// Non-Apple stub for MetalBackend: no Metal, available() == false.
// Compiled instead of MetalBackend.mm on platforms without Metal, so the
// icecube library links everywhere. Callers only construct a MetalBackend after
// checking available(), so these bodies are never reached.

#include "MetalBackend.h"

#include <stdexcept>
#include <utility>

namespace ana::ic {

  MetalBackend::MetalBackend() {
    throw std::runtime_error("MetalBackend: Metal backend not available on this platform");
  }

  MetalBackend::~MetalBackend() = default;

  bool MetalBackend::available() noexcept { return false; }

  std::shared_ptr<GpuSession> MetalBackend::create_session() { return nullptr; }

  std::size_t MetalBackend::column_count() const noexcept { return 0; }
  std::size_t MetalBackend::kernel_compile_count() const noexcept { return 0; }
  std::size_t MetalBackend::live_output_count() const noexcept { return 0; }

  MetalSession::MetalSession(std::shared_ptr<MetalBackend> backend)
    : m_Backend(std::move(backend)) {}

  MetalSession::~MetalSession() = default;

  void MetalSession::ensure_kernel(const char*, const char*) {}
  int  MetalSession::upload_column(const double*, std::size_t) { return -1; }
  int  MetalSession::upload_offsets(const std::size_t*, std::size_t) { return -1; }
  int  MetalSession::alloc_output(std::size_t, bool) { return -1; }
  void MetalSession::dispatch(const char*, const int*, int, const void*, std::size_t, int, int, std::size_t) {}
  const float* MetalSession::contents(int) const noexcept { return nullptr; }

}  // namespace ana::ic
