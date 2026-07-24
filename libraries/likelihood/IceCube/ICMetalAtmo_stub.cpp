// Non-Apple stub for ICMetalAtmo: no Metal, available() == false.
// Compiled instead of ICMetalAtmo.mm on platforms without Metal.

#include "ICMetalAtmo.h"

#include <stdexcept>

namespace ana::ic {

  ICMetalAtmo::ICMetalAtmo(const io::ic::ICSample&, double, double) {
    throw std::runtime_error("ICMetalAtmo: Metal backend not available on this platform");
  }

  ICMetalAtmo::~ICMetalAtmo() = default;

  void ICMetalAtmo::recalculate(double, double, double, double, const double*, bool) {}
  const float* ICMetalAtmo::histogram() const noexcept { return nullptr; }
  const float* ICMetalAtmo::per_event_weight() const noexcept { return nullptr; }
  std::size_t  ICMetalAtmo::size() const noexcept { return 0; }
  bool         ICMetalAtmo::available() noexcept { return false; }

}  // namespace ana::ic
