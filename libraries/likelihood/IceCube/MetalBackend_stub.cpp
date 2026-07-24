// Non-Apple stub for MetalBackend: no Metal, available() == false.
// Compiled instead of MetalBackend.mm on platforms without Metal, so the
// icecube library links everywhere. The flux components only construct a
// MetalBackend after checking available(), so these bodies are never reached.

#include "MetalBackend.h"

#include <stdexcept>

namespace ana::ic {

  MetalBackend::MetalBackend() {
    throw std::runtime_error("MetalBackend: Metal backend not available on this platform");
  }

  MetalBackend::~MetalBackend() = default;

  bool MetalBackend::available() noexcept { return false; }

  void MetalBackend::ensure_kernel(const char*, const char*) {}
  int  MetalBackend::upload_column(const double*, std::size_t) { return -1; }
  int  MetalBackend::upload_offsets(const std::size_t*, std::size_t) { return -1; }
  int  MetalBackend::alloc_output(std::size_t) { return -1; }
  void MetalBackend::dispatch(const char*, const int*, int, const void*, std::size_t, int, int) {}
  const float* MetalBackend::contents(int) const noexcept { return nullptr; }

}  // namespace ana::ic
