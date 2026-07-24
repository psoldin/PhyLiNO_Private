// Non-Apple stub for ICMetalPowerlaw: no Metal, available() == false.
// Compiled instead of ICMetalPowerlaw.mm on platforms without Metal, so the
// icecube library links everywhere while PowerlawFlux gates on available().

#include "ICMetalPowerlaw.h"

#include <stdexcept>

namespace ana::ic {

  ICMetalPowerlaw::ICMetalPowerlaw(const io::ic::ICSample&, double, double, bool) {
    throw std::runtime_error("ICMetalPowerlaw: Metal backend not available on this platform");
  }

  ICMetalPowerlaw::~ICMetalPowerlaw() = default;

  void         ICMetalPowerlaw::recalculate(double, double, bool) {}
  const float* ICMetalPowerlaw::histogram() const noexcept { return nullptr; }
  const float* ICMetalPowerlaw::per_event_weight() const noexcept { return nullptr; }
  std::size_t  ICMetalPowerlaw::size() const noexcept { return 0; }
  bool         ICMetalPowerlaw::available() noexcept { return false; }

}  // namespace ana::ic
