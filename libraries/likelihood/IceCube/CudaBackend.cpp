// CUDA implementation of GpuBackend. Uses the CUDA driver API + NVRTC so kernels
// are compiled at runtime from the CUDA-C source strings the flux components
// ship alongside their MSL strings. No device code is compiled by nvcc here --
// NVRTC does it at runtime -- so this is an ordinary .cpp built only when the
// CUDA toolkit is found (guarded in CMakeLists.txt); a stub replaces it
// otherwise.

#include "CudaBackend.h"

#include "../../io/IceCube/ICConstants.h"

#include <cuda.h>
#include <nvrtc.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

    // One device allocation plus, for outputs, a host mirror dispatch() refreshes
    // so contents() can hand back a CPU pointer like the Metal shared-memory path.
    // Scalar columns/outputs are 4 B (float) in FP32 mode or 8 B (double) in FP64
    // mode; CSR offsets are always 4 B (uint32). Only the host mirror matching the
    // active precision is populated.
    struct Buffer {
      CUdeviceptr         dptr       = 0;
      std::size_t         n_elems    = 0;  // element count
      std::size_t         elem_bytes = 4;  // 4 = float/uint32, 8 = double
      std::vector<float>  host32;          // output mirror in FP32 mode
      std::vector<double> host64;          // output mirror in FP64 mode
    };

    struct CudaState {
      CUdevice  dev  = 0;
      CUcontext ctx  = nullptr;
      bool      fp64 = false;  // FP64 compute path when true

      std::vector<Buffer>                         buffers;   // handle -> buffer
      std::unordered_map<const void*, int>        colCache;  // source ptr -> handle
      std::unordered_map<std::string, CUfunction> funcs;     // kernel name -> fn
      std::vector<CUmodule>                       modules;   // kept alive for funcs

      // Byte size of a scalar column/output element in the active precision.
      [[nodiscard]] std::size_t scalar_bytes() const noexcept { return fp64 ? 8 : 4; }
    };

  }  // namespace

  CudaBackend::CudaBackend(bool fp64) : m_Fp64(fp64) {
    cu_check(cuInit(0), "cuInit");
    int count = 0;
    cu_check(cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (count == 0)
      throw std::runtime_error("CudaBackend: no CUDA device available");

    auto* s  = new CudaState;
    s->fp64  = fp64;
    try {
      cu_check(cuDeviceGet(&s->dev, 0), "cuDeviceGet");
      cu_check(cuCtxCreate(&s->ctx, 0, s->dev), "cuCtxCreate");
    } catch (...) {
      delete s;
      throw;
    }
    m_State = s;
  }

  CudaBackend::~CudaBackend() {
    auto* s = static_cast<CudaState*>(m_State);
    if (!s) return;
    for (auto& b : s->buffers)
      if (b.dptr) cuMemFree(b.dptr);
    for (CUmodule m : s->modules) cuModuleUnload(m);
    if (s->ctx) cuCtxDestroy(s->ctx);
    delete s;
  }

  bool CudaBackend::available() noexcept {
    if (cuInit(0) != CUDA_SUCCESS) return false;
    int count = 0;
    return cuDeviceGetCount(&count) == CUDA_SUCCESS && count > 0;
  }

  void CudaBackend::ensure_kernel(const char* name, const char* source) {
    auto* s = static_cast<CudaState*>(m_State);
    const std::string key(name);
    if (s->funcs.count(key)) return;

    // Compile to PTX for the device's compute capability; the driver JITs it to
    // SASS at module load (mirrors Metal newLibraryWithSource -> pipeline state).
    int major = 0, minor = 0;
    cu_check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, s->dev),
             "compute-capability major");
    cu_check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, s->dev),
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

    s->modules.push_back(mod);
    s->funcs[key] = fn;
  }

  int CudaBackend::upload_column(const double* data, std::size_t n) {
    auto* s = static_cast<CudaState*>(m_State);
    if (auto it = s->colCache.find(data); it != s->colCache.end())
      return it->second;

    Buffer b;
    b.n_elems    = n;
    b.elem_bytes = s->scalar_bytes();
    cu_check(cuMemAlloc(&b.dptr, n * b.elem_bytes), "cuMemAlloc(column)");
    if (s->fp64) {
      // Already double: upload straight from the source column, no conversion.
      cu_check(cuMemcpyHtoD(b.dptr, data, n * sizeof(double)), "cuMemcpyHtoD(column)");
    } else {
      std::vector<float> f(n);
      for (std::size_t i = 0; i < n; ++i) f[i] = static_cast<float>(data[i]);
      cu_check(cuMemcpyHtoD(b.dptr, f.data(), n * sizeof(float)), "cuMemcpyHtoD(column)");
    }

    const int handle = static_cast<int>(s->buffers.size());
    s->buffers.push_back(std::move(b));
    s->colCache[data] = handle;
    return handle;
  }

  int CudaBackend::upload_offsets(const std::size_t* data, std::size_t n) {
    auto* s = static_cast<CudaState*>(m_State);
    if (auto it = s->colCache.find(data); it != s->colCache.end())
      return it->second;

    std::vector<std::uint32_t> u(n);
    for (std::size_t i = 0; i < n; ++i) u[i] = static_cast<std::uint32_t>(data[i]);

    Buffer b;
    b.n_elems    = n;
    b.elem_bytes = sizeof(std::uint32_t);  // offsets stay uint32 in both precisions
    cu_check(cuMemAlloc(&b.dptr, n * sizeof(std::uint32_t)), "cuMemAlloc(offsets)");
    cu_check(cuMemcpyHtoD(b.dptr, u.data(), n * sizeof(std::uint32_t)), "cuMemcpyHtoD(offsets)");

    const int handle = static_cast<int>(s->buffers.size());
    s->buffers.push_back(std::move(b));
    s->colCache[data] = handle;
    return handle;
  }

  int CudaBackend::alloc_output(std::size_t n) {
    auto* s = static_cast<CudaState*>(m_State);
    Buffer b;
    b.n_elems    = n;
    b.elem_bytes = s->scalar_bytes();
    if (s->fp64) b.host64.assign(n, 0.0);
    else         b.host32.assign(n, 0.0f);
    cu_check(cuMemAlloc(&b.dptr, n * b.elem_bytes), "cuMemAlloc(output)");
    cu_check(cuMemsetD8(b.dptr, 0, n * b.elem_bytes), "cuMemsetD8(output)");

    const int handle = static_cast<int>(s->buffers.size());
    s->buffers.push_back(std::move(b));
    return handle;
  }

  void CudaBackend::dispatch(const char* name,
                             const int*  inputs,
                             const int   n_inputs,
                             const void* params,
                             std::size_t /*params_len*/,
                             const int   hist,
                             const int   per_event,
                             const std::size_t n_groups) {
    auto* s = static_cast<CudaState*>(m_State);
    // CUDA driver contexts are thread-local; dispatch() may run on a worker
    // thread (ICLikelihood computes samples concurrently), so bind the backend's
    // context here. Idempotent and cheap when it is already current.
    cu_check(cuCtxSetCurrent(s->ctx), "cuCtxSetCurrent");
    CUfunction fn = s->funcs.at(std::string(name));

    // cuLaunchKernel wants an array of pointers to each argument. Kernel
    // signature order matches the buffer convention: inputs..., params (by
    // value), hist, per_event. Keep every pointer arg in a stable lvalue.
    std::vector<CUdeviceptr> dptrs;
    dptrs.reserve(static_cast<std::size_t>(n_inputs));
    for (int i = 0; i < n_inputs; ++i) dptrs.push_back(s->buffers[inputs[i]].dptr);
    CUdeviceptr hist_ptr = s->buffers[hist].dptr;
    CUdeviceptr pe_ptr   = per_event >= 0 ? s->buffers[per_event].dptr : 0;

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
                            0, nullptr, args.data(), nullptr),
             "cuLaunchKernel");
    cu_check(cuCtxSynchronize(), "cuCtxSynchronize");

    // Refresh host mirrors of the outputs so contents()/contents_f64() return a
    // CPU pointer. The per-event copy is skipped by the caller passing
    // per_event < 0 when the weights are not needed (e.g. the Poisson path).
    auto refresh = [&](int handle) {
      Buffer& b = s->buffers[handle];
      void* dst = s->fp64 ? static_cast<void*>(b.host64.data())
                          : static_cast<void*>(b.host32.data());
      cu_check(cuMemcpyDtoH(dst, b.dptr, b.n_elems * b.elem_bytes), "cuMemcpyDtoH(output)");
    };
    refresh(hist);
    if (per_event >= 0) refresh(per_event);
  }

  const float* CudaBackend::contents(int handle) const noexcept {
    auto* s = static_cast<CudaState*>(m_State);
    return s->buffers[handle].host32.data();
  }

  const double* CudaBackend::contents_f64(int handle) const noexcept {
    auto* s = static_cast<CudaState*>(m_State);
    return s->buffers[handle].host64.data();
  }

}  // namespace ana::ic
