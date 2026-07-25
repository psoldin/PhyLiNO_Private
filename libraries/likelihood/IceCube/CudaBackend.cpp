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
    struct Buffer {
      CUdeviceptr        dptr    = 0;
      std::size_t        n_elems = 0;  // element count (float or uint32, both 4 B)
      std::vector<float> host;         // non-empty only for outputs
    };

    struct CudaState {
      CUdevice  dev = 0;
      CUcontext ctx = nullptr;

      std::vector<Buffer>                         buffers;   // handle -> buffer
      std::unordered_map<const void*, int>        colCache;  // source ptr -> handle
      std::unordered_map<std::string, CUfunction> funcs;     // kernel name -> fn
      std::vector<CUmodule>                       modules;   // kept alive for funcs
    };

  }  // namespace

  CudaBackend::CudaBackend() {
    cu_check(cuInit(0), "cuInit");
    int count = 0;
    cu_check(cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (count == 0)
      throw std::runtime_error("CudaBackend: no CUDA device available");

    auto* s = new CudaState;
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

    std::vector<float> f(n);
    for (std::size_t i = 0; i < n; ++i) f[i] = static_cast<float>(data[i]);

    Buffer b;
    b.n_elems = n;
    cu_check(cuMemAlloc(&b.dptr, n * sizeof(float)), "cuMemAlloc(column)");
    cu_check(cuMemcpyHtoD(b.dptr, f.data(), n * sizeof(float)), "cuMemcpyHtoD(column)");

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
    b.n_elems = n;
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
    b.n_elems = n;
    b.host.assign(n, 0.0f);
    cu_check(cuMemAlloc(&b.dptr, n * sizeof(float)), "cuMemAlloc(output)");
    cu_check(cuMemsetD8(b.dptr, 0, n * sizeof(float)), "cuMemsetD8(output)");

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
    if (per_event >= 0) args.push_back(&pe_ptr);         // optional per-event

    cu_check(cuLaunchKernel(fn,
                            static_cast<unsigned>(n_groups), 1, 1,
                            kThreadsPerGroup, 1, 1,
                            0, nullptr, args.data(), nullptr),
             "cuLaunchKernel");
    cu_check(cuCtxSynchronize(), "cuCtxSynchronize");

    // Refresh host mirrors of the outputs so contents() returns a CPU pointer.
    // The per-event copy is skipped by the caller passing per_event < 0 when the
    // weights are not needed (e.g. the Poisson path).
    Buffer& h = s->buffers[hist];
    cu_check(cuMemcpyDtoH(h.host.data(), h.dptr, h.n_elems * sizeof(float)), "cuMemcpyDtoH(hist)");
    if (per_event >= 0) {
      Buffer& pe = s->buffers[per_event];
      cu_check(cuMemcpyDtoH(pe.host.data(), pe.dptr, pe.n_elems * sizeof(float)), "cuMemcpyDtoH(pe)");
    }
  }

  const float* CudaBackend::contents(int handle) const noexcept {
    auto* s = static_cast<CudaState*>(m_State);
    return s->buffers[handle].host.data();
  }

}  // namespace ana::ic
