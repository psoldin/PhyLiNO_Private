// Non-CUDA stub for CudaBackend: no CUDA toolkit, available() == false.
// Compiled instead of CudaBackend.cpp on platforms/builds without CUDA, so the
// icecube library links everywhere. make_gpu_backend() only constructs a
// CudaBackend after checking available(), so these bodies are never reached.

#include "CudaBackend.h"

#include <stdexcept>

namespace ana::ic {

  CudaBackend::CudaBackend(bool) {
    throw std::runtime_error("CudaBackend: CUDA backend not available in this build");
  }

  CudaBackend::~CudaBackend() = default;

  bool CudaBackend::available() noexcept { return false; }

  void CudaBackend::ensure_kernel(const char*, const char*) {}
  int  CudaBackend::upload_column(const double*, std::size_t) { return -1; }
  int  CudaBackend::upload_offsets(const std::size_t*, std::size_t) { return -1; }
  int  CudaBackend::alloc_output(std::size_t) { return -1; }
  void CudaBackend::dispatch(const char*, const int*, int, const void*, std::size_t, int, int, std::size_t) {}
  const float*  CudaBackend::contents(int) const noexcept { return nullptr; }
  const double* CudaBackend::contents_f64(int) const noexcept { return nullptr; }

}  // namespace ana::ic
