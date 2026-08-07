// Non-CUDA stub for CudaBackend: no CUDA toolkit, available() == false.
// Compiled instead of CudaBackend.cpp on platforms/builds without CUDA, so the
// icecube library links everywhere. make_gpu_backend() only constructs a
// CudaBackend after checking available(), so these bodies are never reached.

#include "CudaBackend.h"

#include <stdexcept>
#include <utility>

namespace ana::ic {

  CudaBackend::CudaBackend(bool) {
    throw std::runtime_error("CudaBackend: CUDA backend not available in this build");
  }

  CudaBackend::~CudaBackend() = default;

  bool CudaBackend::available() noexcept { return false; }

  std::shared_ptr<GpuSession> CudaBackend::create_session() { return nullptr; }

  std::size_t CudaBackend::column_count() const noexcept { return 0; }
  std::size_t CudaBackend::kernel_compile_count() const noexcept { return 0; }
  std::size_t CudaBackend::live_output_count() const noexcept { return 0; }

  CudaSession::CudaSession(std::shared_ptr<CudaBackend> backend)
    : m_Backend(std::move(backend)) {}

  CudaSession::~CudaSession() = default;

  bool CudaSession::is_fp64() const noexcept { return false; }

  void CudaSession::ensure_kernel(const char*, const char*) {}
  int  CudaSession::upload_column(const double*, std::size_t) { return -1; }
  int  CudaSession::upload_offsets(const std::size_t*, std::size_t) { return -1; }
  int  CudaSession::alloc_output(std::size_t, bool) { return -1; }
  void CudaSession::dispatch(const char*, const int*, int, const void*, std::size_t, int, int, std::size_t) {}
  const float*  CudaSession::contents(int) const noexcept { return nullptr; }
  const double* CudaSession::contents_f64(int) const noexcept { return nullptr; }

}  // namespace ana::ic
