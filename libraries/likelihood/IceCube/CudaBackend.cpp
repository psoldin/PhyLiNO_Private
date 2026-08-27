// CUDA implementation of GpuBackend. Uses the CUDA driver API + NVRTC so kernels
// are compiled at runtime from the CUDA-C source strings the flux components
// ship alongside their MSL strings. No device code is compiled by nvcc here --
// NVRTC does it at runtime -- so this is an ordinary .cpp built only when the
// CUDA toolkit is found (guarded in CMakeLists.txt); a stub replaces it
// otherwise.
//
// Split of state: CudaState is process-wide and holds what is identical for
// every sample and every fit (context, compiled modules, uploaded MC columns);
// CudaSessionState holds one sample's output buffers and its handle table.

#include "CudaBackend.h"

#include "../../io/IceCube/ICConstants.h"

#include <cuda.h>
#include <nvrtc.h>

#include <atomic>
#include <cstdint>
#include <deque>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ana::ic {

  namespace {

    constexpr unsigned int kThreadsPerGroup = 256;  // must match every kernel

    void cu_check(CUresult r, const char* what) {
      if (r != CUDA_SUCCESS) {
        const char* msg = nullptr;
        cuGetErrorString(r, &msg);
        throw std::runtime_error(std::string("CudaBackend: ") + what + ": " +
                                 (msg ? msg : "unknown error"));
      }
    }

    void nvrtc_check(nvrtcResult r, const char* what) {
      if (r != NVRTC_SUCCESS)
        throw std::runtime_error(std::string("CudaBackend: ") + what + ": " +
                                 nvrtcGetErrorString(r));
    }

    // A device allocation the backend owns: one uploaded MC column, shared by
    // every session. Scalar columns are 4 B (float) in FP32 mode or 8 B (double)
    // in FP64 mode; CSR offsets are always 4 B (uint32).
    struct Column {
      CUdeviceptr dptr    = 0;
      std::size_t n_elems = 0;
    };

    // One row of a session's flat handle table. An owning row is an output this
    // session allocated and must free; a non-owning row aliases a backend column
    // so that callers see one flat handle space (see GpuSession). Only owning
    // rows *declared readable* carry a host mirror -- columns are never read
    // back, and neither are the outputs that exist purely to feed the next
    // kernel (see alloc_output).
    struct SessionRow {
      CUdeviceptr         dptr       = 0;
      std::size_t         n_elems    = 0;
      std::size_t         elem_bytes = 4;  // 4 = float/uint32, 8 = double
      bool                owning     = false;
      bool                readback   = false;
      std::vector<float>  host32;          // output mirror in FP32 mode
      std::vector<double> host64;          // output mirror in FP64 mode
    };

    struct CudaState {
      CUdevice  dev  = 0;
      CUcontext ctx  = nullptr;
      bool      fp64 = false;  // FP64 compute path when true

      // Guards everything below. Scan workers build their Fits -- and therefore
      // their sessions -- concurrently, so the warmup paths that populate these
      // run on several threads at once. The hot path (dispatch) takes no lock:
      // it only reads funcs, which is stable once every kernel is compiled, and
      // its own session's rows.
      //
      // shared_mutex, not mutex: a scan builds one Fit per grid point, so
      // ensure_kernel/upload_column/upload_offsets are each called thousands of
      // times but *populate* these containers only on the first. Every later
      // call is a pure cache hit, and taking the exclusive lock for it
      // serialised the whole warmup across every scan worker. Lookups take the
      // shared lock; only the population path (and the NVRTC compile it holds
      // across) takes the exclusive one.
      std::shared_mutex mutex;

      // deque, not vector: dispatch reads device pointers out of a session's
      // alias rows while another thread may still be appending columns, and a
      // reallocation would move the elements under it.
      std::deque<Column>                          columns;   // shared column allocations
      std::unordered_map<const void*, int>        colCache;  // source ptr -> index into columns
      std::unordered_map<std::string, CUfunction> funcs;     // kernel name -> fn
      std::vector<CUmodule>                       modules;   // kept alive for funcs

      // Test-visible counters (see GpuBackend). Atomic rather than derived from
      // the containers above, so reading one needs no lock and cannot throw.
      std::atomic<std::size_t> columnCount{0};
      std::atomic<std::size_t> kernelCount{0};
      std::atomic<std::size_t> liveOutputs{0};

      // Byte size of a scalar column/output element in the active precision.
      [[nodiscard]] std::size_t scalar_bytes() const noexcept { return fp64 ? 8 : 4; }
    };

    struct CudaSessionState {
      // deque so that contents()'s pointer into a row's host mirror survives a
      // later alloc_output on the same session.
      std::deque<SessionRow> rows;

      // This sample's own stream. Samples are evaluated concurrently and scan
      // workers run whole fits concurrently, so the launches must be able to
      // overlap; synchronising per stream is what allows that.
      CUstream stream = nullptr;
    };

    // Register a shared backend column in this session's table as a non-owning
    // alias, so callers see one flat handle space (see GpuSession).
    int alias_row(CudaSessionState* s, const Column& col, std::size_t elem_bytes) {
      SessionRow row;
      row.dptr       = col.dptr;
      row.n_elems    = col.n_elems;
      row.elem_bytes = elem_bytes;
      row.owning     = false;

      const int handle = static_cast<int>(s->rows.size());
      s->rows.push_back(std::move(row));
      return handle;
    }

  }  // namespace

  CudaBackend::CudaBackend(bool fp64, int device) : m_Fp64(fp64) {
    cu_check(cuInit(0), "cuInit");
    int count = 0;
    cu_check(cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (count == 0)
      throw std::runtime_error("CudaBackend: no CUDA device available");
    if (device < 0 || device >= count)
      throw std::runtime_error("CudaBackend: device ordinal " + std::to_string(device) +
                               " out of range (" + std::to_string(count) + " visible)");

    auto* s  = new CudaState;
    s->fp64  = fp64;
    try {
      cu_check(cuDeviceGet(&s->dev, device), "cuDeviceGet");
      // The device's primary context, not a fresh one: a context created with
      // cuCtxCreate is current only on the creating thread, and this backend is
      // shared across the scan workers that build sessions and dispatch on
      // them. The primary context can be made current on any of them.
      cu_check(cuDevicePrimaryCtxRetain(&s->ctx, s->dev), "cuDevicePrimaryCtxRetain");
      cu_check(cuCtxSetCurrent(s->ctx), "cuCtxSetCurrent");
    } catch (...) {
      delete s;
      throw;
    }
    m_State = s;
  }

  CudaBackend::~CudaBackend() {
    auto* s = static_cast<CudaState*>(m_State);
    if (!s) return;
    if (s->ctx) cuCtxSetCurrent(s->ctx);
    for (auto& c : s->columns)
      if (c.dptr) cuMemFree(c.dptr);
    for (CUmodule m : s->modules) cuModuleUnload(m);
    if (s->ctx) cuDevicePrimaryCtxRelease(s->dev);
    delete s;
  }

  bool CudaBackend::available() noexcept {
    if (cuInit(0) != CUDA_SUCCESS) return false;
    int count = 0;
    return cuDeviceGetCount(&count) == CUDA_SUCCESS && count > 0;
  }

  std::shared_ptr<GpuSession> CudaBackend::create_session() {
    return std::make_shared<CudaSession>(
        std::static_pointer_cast<CudaBackend>(shared_from_this()));
  }

  std::size_t CudaBackend::column_count() const noexcept {
    auto* s = static_cast<CudaState*>(m_State);
    return s->columnCount.load(std::memory_order_relaxed);
  }

  std::size_t CudaBackend::kernel_compile_count() const noexcept {
    auto* s = static_cast<CudaState*>(m_State);
    return s->kernelCount.load(std::memory_order_relaxed);
  }

  std::size_t CudaBackend::live_output_count() const noexcept {
    auto* s = static_cast<CudaState*>(m_State);
    return s->liveOutputs.load(std::memory_order_relaxed);
  }

  CudaSession::CudaSession(std::shared_ptr<CudaBackend> backend)
    : m_Backend(std::move(backend)) {
    if (!m_Backend)
      throw std::runtime_error("CudaSession: null backend");

    auto* b = static_cast<CudaState*>(m_Backend->m_State);
    auto* s = new CudaSessionState;
    try {
      cu_check(cuCtxSetCurrent(b->ctx), "cuCtxSetCurrent");
      // CU_STREAM_NON_BLOCKING, not the default flags: a default-flag stream
      // implicitly synchronises against the legacy default stream, which would
      // serialise every session against every other one and defeat the point.
      cu_check(cuStreamCreate(&s->stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    } catch (...) {
      delete s;
      // Deliberately fatal rather than falling back to the CPU path for this one
      // sample: scan points that silently used different code paths would not be
      // comparable with each other.
      throw;
    }
    m_State = s;
  }

  CudaSession::~CudaSession() {
    auto* s = static_cast<CudaSessionState*>(m_State);
    if (!s) return;
    auto* b = static_cast<CudaState*>(m_Backend->m_State);
    if (b && b->ctx) cuCtxSetCurrent(b->ctx);
    // Free only what this session allocated: alias rows point at backend columns
    // that other sessions are still using.
    for (auto& row : s->rows) {
      if (!row.owning) continue;
      if (row.dptr) cuMemFree(row.dptr);
      if (b) b->liveOutputs.fetch_sub(1, std::memory_order_relaxed);
    }
    if (s->stream) cuStreamDestroy(s->stream);
    delete s;
  }

  bool CudaSession::is_fp64() const noexcept { return m_Backend->is_fp64(); }

  void CudaSession::ensure_kernel(const char* name, const char* source) {
    auto* b = static_cast<CudaState*>(m_Backend->m_State);
    const std::string key(name);

    // Already compiled: a shared lock and nothing else. Every Fit's components
    // re-request their kernels, so all but the first of thousands of calls end
    // here.
    {
      const std::shared_lock lock(b->mutex);
      if (b->funcs.count(key)) return;
    }

    // Exclusive, and held across the NVRTC compile: concurrent sessions asking
    // for the same kernel must not each compile it. Only the first pays; the
    // compile happens once per process for each of the four kernels.
    const std::scoped_lock lock(b->mutex);
    cu_check(cuCtxSetCurrent(b->ctx), "cuCtxSetCurrent");

    // Re-check: another worker may have compiled it between the two locks.
    if (b->funcs.count(key)) return;

    // Compile to PTX for the device's compute capability; the driver JITs it to
    // SASS at module load (mirrors Metal newLibraryWithSource -> pipeline state).
    int major = 0, minor = 0;
    cu_check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, b->dev),
             "compute-capability major");
    cu_check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, b->dev),
             "compute-capability minor");
    const std::string arch =
        "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);

    nvrtcProgram prog = nullptr;
    nvrtc_check(nvrtcCreateProgram(&prog, source, (key + ".cu").c_str(), 0, nullptr, nullptr),
                "nvrtcCreateProgram");

    const char*        opts[] = {arch.c_str(), "--std=c++17"};
    const nvrtcResult  cres   = nvrtcCompileProgram(prog, 2, opts);
    if (cres != NVRTC_SUCCESS) {
      std::size_t log_size = 0;
      nvrtcGetProgramLogSize(prog, &log_size);
      std::string log(log_size, '\0');
      nvrtcGetProgramLog(prog, log.data());
      nvrtcDestroyProgram(&prog);
      throw std::runtime_error(std::string("CudaBackend: compile of '") + name + "' failed:\n" + log);
    }

    std::size_t ptx_size = 0;
    nvrtc_check(nvrtcGetPTXSize(prog, &ptx_size), "nvrtcGetPTXSize");
    std::string ptx(ptx_size, '\0');
    nvrtc_check(nvrtcGetPTX(prog, ptx.data()), "nvrtcGetPTX");
    nvrtcDestroyProgram(&prog);

    CUmodule mod = nullptr;
    cu_check(cuModuleLoadData(&mod, ptx.c_str()), "cuModuleLoadData");
    CUfunction fn = nullptr;
    cu_check(cuModuleGetFunction(&fn, mod, name), "cuModuleGetFunction");  // needs extern "C"

    b->modules.push_back(mod);
    b->funcs[key] = fn;
    b->kernelCount.fetch_add(1, std::memory_order_relaxed);
  }

  int CudaSession::upload_column(const double* data, std::size_t n) {
    auto* b = static_cast<CudaState*>(m_Backend->m_State);
    auto* s = static_cast<CudaSessionState*>(m_State);

    // Cache hit under a shared lock and with no CUDA call at all: registering
    // the alias only touches this session's own rows, and cuCtxSetCurrent is
    // needed for the allocation below, not for the lookup.
    {
      const std::shared_lock lock(b->mutex);
      if (auto it = b->colCache.find(data); it != b->colCache.end())
        return alias_row(s, b->columns[it->second], b->scalar_bytes());
    }

    const std::scoped_lock lock(b->mutex);
    cu_check(cuCtxSetCurrent(b->ctx), "cuCtxSetCurrent");

    // Re-check: another worker may have uploaded this column while we were
    // between the two locks.
    if (auto it = b->colCache.find(data); it != b->colCache.end())
      return alias_row(s, b->columns[it->second], b->scalar_bytes());

    Column c;
    c.n_elems = n;
    cu_check(cuMemAlloc(&c.dptr, n * b->scalar_bytes()), "cuMemAlloc(column)");
    if (b->fp64) {
      // Already double: upload straight from the source column, no conversion.
      cu_check(cuMemcpyHtoD(c.dptr, data, n * sizeof(double)), "cuMemcpyHtoD(column)");
    } else {
      std::vector<float> f(n);
      for (std::size_t i = 0; i < n; ++i) f[i] = static_cast<float>(data[i]);
      cu_check(cuMemcpyHtoD(c.dptr, f.data(), n * sizeof(float)), "cuMemcpyHtoD(column)");
    }

    b->colCache[data] = static_cast<int>(b->columns.size());
    b->columns.push_back(c);
    b->columnCount.fetch_add(1, std::memory_order_relaxed);
    return alias_row(s, c, b->scalar_bytes());
  }

  int CudaSession::upload_offsets(const std::size_t* data, std::size_t n) {
    auto* b = static_cast<CudaState*>(m_Backend->m_State);
    auto* s = static_cast<CudaSessionState*>(m_State);

    // offsets stay uint32 in both precisions
    constexpr std::size_t kOffsetBytes = sizeof(std::uint32_t);

    // Shared-lock cache hit, exclusive only to upload -- see upload_column.
    {
      const std::shared_lock lock(b->mutex);
      if (auto it = b->colCache.find(data); it != b->colCache.end())
        return alias_row(s, b->columns[it->second], kOffsetBytes);
    }

    const std::scoped_lock lock(b->mutex);
    cu_check(cuCtxSetCurrent(b->ctx), "cuCtxSetCurrent");

    if (auto it = b->colCache.find(data); it != b->colCache.end())
      return alias_row(s, b->columns[it->second], kOffsetBytes);

    std::vector<std::uint32_t> u(n);
    for (std::size_t i = 0; i < n; ++i) u[i] = static_cast<std::uint32_t>(data[i]);

    Column c;
    c.n_elems = n;
    cu_check(cuMemAlloc(&c.dptr, n * kOffsetBytes), "cuMemAlloc(offsets)");
    cu_check(cuMemcpyHtoD(c.dptr, u.data(), n * kOffsetBytes), "cuMemcpyHtoD(offsets)");

    b->colCache[data] = static_cast<int>(b->columns.size());
    b->columns.push_back(c);
    b->columnCount.fetch_add(1, std::memory_order_relaxed);
    return alias_row(s, c, kOffsetBytes);
  }

  int CudaSession::alloc_output(std::size_t n, const bool readback) {
    auto* b = static_cast<CudaState*>(m_Backend->m_State);
    auto* s = static_cast<CudaSessionState*>(m_State);

    // No lock: the row goes into this session's own table. The context still has
    // to be bound, because the allocation happens on whichever worker thread
    // built this fit.
    cu_check(cuCtxSetCurrent(b->ctx), "cuCtxSetCurrent");

    SessionRow row;
    row.n_elems    = n;
    row.elem_bytes = b->scalar_bytes();
    row.owning     = true;
    row.readback   = readback;
    // The mirror is only allocated for a buffer someone will actually read. The
    // per-event buffers are event-sized (>10^7 for the tracks sample), so
    // mirroring them unconditionally cost a nine-figure host allocation per
    // session on top of the copy that filled it.
    if (readback) {
      if (b->fp64) row.host64.assign(n, 0.0);
      else         row.host32.assign(n, 0.0f);
    }
    cu_check(cuMemAlloc(&row.dptr, n * row.elem_bytes), "cuMemAlloc(output)");
    cu_check(cuMemsetD8(row.dptr, 0, n * row.elem_bytes), "cuMemsetD8(output)");

    const int handle = static_cast<int>(s->rows.size());
    s->rows.push_back(std::move(row));
    b->liveOutputs.fetch_add(1, std::memory_order_relaxed);
    return handle;
  }

  void CudaSession::dispatch(const char* name,
                             const int*  inputs,
                             const int   n_inputs,
                             const void* params,
                             std::size_t /*params_len*/,
                             const int   hist,
                             const int   per_event,
                             const std::size_t n_groups) {
    auto* b = static_cast<CudaState*>(m_Backend->m_State);
    auto* s = static_cast<CudaSessionState*>(m_State);
    // CUDA driver contexts are thread-local; dispatch() may run on a worker
    // thread (ICLikelihood computes samples concurrently), so bind the backend's
    // context here. Idempotent and cheap when it is already current.
    cu_check(cuCtxSetCurrent(b->ctx), "cuCtxSetCurrent");
    CUfunction fn = b->funcs.at(std::string(name));

    // cuLaunchKernel wants an array of pointers to each argument. Kernel
    // signature order matches the buffer convention: inputs..., params (by
    // value), hist, per_event. Keep every pointer arg in a stable lvalue.
    std::vector<CUdeviceptr> dptrs;
    dptrs.reserve(static_cast<std::size_t>(n_inputs));
    for (int i = 0; i < n_inputs; ++i) dptrs.push_back(s->rows[inputs[i]].dptr);
    CUdeviceptr hist_ptr = s->rows[hist].dptr;
    CUdeviceptr pe_ptr   = per_event >= 0 ? s->rows[per_event].dptr : 0;

    std::vector<void*> args;
    args.reserve(static_cast<std::size_t>(n_inputs) + 3);
    for (auto& d : dptrs) args.push_back(&d);           // input device pointers
    args.push_back(const_cast<void*>(params));          // params struct, by value
    args.push_back(&hist_ptr);                           // histogram
    // The compiled kernel always declares a per_event parameter (write_pe gates
    // whether it's touched), so the argument must always be present -- pe_ptr is
    // 0 when per_event < 0, which the kernel never dereferences in that case.
    args.push_back(&pe_ptr);                             // per-event (maybe unused)

    cu_check(cuLaunchKernel(fn,
                            static_cast<unsigned>(n_groups), 1, 1,
                            kThreadsPerGroup, 1, 1,
                            0, s->stream, args.data(), nullptr),
             "cuLaunchKernel");

    // Queue the host-mirror refresh behind the launch on the same stream, so
    // contents()/contents_f64() can return a CPU pointer.
    //
    // Only buffers allocated with readback=true are copied. The per-event
    // weights deliberately are not: they exist to be read by the *next* kernel
    // (say_ssq) and no caller ever asks for their host mirror, so copying them
    // was moving one double per MC event per dispatch back over PCIe for
    // nothing -- ~220 MB per likelihood evaluation on the 3D tracks sample, in
    // front of the synchronise below.
    auto refresh = [&](int handle) {
      SessionRow& row = s->rows[handle];
      if (!row.readback) return;
      void* dst = b->fp64 ? static_cast<void*>(row.host64.data())
                          : static_cast<void*>(row.host32.data());
      cu_check(cuMemcpyDtoHAsync(dst, row.dptr, row.n_elems * row.elem_bytes, s->stream),
               "cuMemcpyDtoHAsync(output)");
    };
    refresh(hist);
    if (per_event >= 0) refresh(per_event);

    // Per stream, not cuCtxSynchronize: waiting on the whole context would block
    // on every other session's work too, which is exactly the serialisation the
    // streams exist to avoid.
    cu_check(cuStreamSynchronize(s->stream), "cuStreamSynchronize");
  }

  const float* CudaSession::contents(int handle) const noexcept {
    auto* s = static_cast<CudaSessionState*>(m_State);
    return s->rows[handle].host32.data();
  }

  const double* CudaSession::contents_f64(int handle) const noexcept {
    auto* s = static_cast<CudaSessionState*>(m_State);
    return s->rows[handle].host64.data();
  }

}  // namespace ana::ic
