#pragma once

#include "../../io/IceCube/ICSample.h"

#include <cstddef>

namespace ana::ic {

  /**
   * Metal (Apple GPU) backend for the astrophysical power-law flux histogram.
   *
   * Pure-C++ facade over an Obj-C++/Metal implementation (see ICMetalPowerlaw.mm);
   * the Metal and Objective-C types never appear in this header, so the rest of
   * the likelihood tree stays plain C++. On non-Apple builds a stub
   * (ICMetalPowerlaw_stub.cpp) provides the same symbols with available()==false.
   *
   * The per-event MC columns (e_true, astro_baseline, CSR bin_offsets) are copied
   * to FP32 GPU buffers ONCE at construction; each recalculate() only pushes the
   * three scalars that change between minimizer iterations. Results (per-bin
   * histogram, and optionally per-event weights) live in shared/unified memory and
   * are read back through histogram()/per_event_weight() with no explicit copy.
   *
   * Precision: FP32 weights + threadgroup tree reduction, validated against the
   * FP64 CPU path to ~5e-7 per bin (far below Poisson bin noise).
   */
  class ICMetalPowerlaw {
   public:
    ICMetalPowerlaw(const io::ic::ICSample& sample,
                    double                  e_ref_gev,
                    double                  reference_index,
                    bool                    per_type_norm);
    ~ICMetalPowerlaw();

    ICMetalPowerlaw(const ICMetalPowerlaw&)            = delete;
    ICMetalPowerlaw& operator=(const ICMetalPowerlaw&) = delete;

    /**
     * Dispatch the powerlaw kernel for (AstroNorm=norm, SpectralIndex=gamma).
     * per-event weights are only written when fill_per_event is true (needed by
     * the SAY fluctuation term; skipped for the Poisson likelihood).
     */
    void recalculate(double norm, double gamma, bool fill_per_event);

    /** Per-bin histogram, io::ic::Constants::nBins floats. Valid after recalculate(). */
    [[nodiscard]] const float* histogram() const noexcept;

    /** Per-event weights in CSR bin order, size() floats. Valid only after a
        recalculate(..., fill_per_event=true). */
    [[nodiscard]] const float* per_event_weight() const noexcept;

    /** Number of in-range events (== sample.size()). */
    [[nodiscard]] std::size_t size() const noexcept;

    /** True if a usable Metal device is present. Cheap; safe to call anywhere. */
    [[nodiscard]] static bool available() noexcept;

   private:
    void* m_State = nullptr;  // opaque MetalState* (Obj-C++), or nullptr in the stub
  };

}  // namespace ana::ic
