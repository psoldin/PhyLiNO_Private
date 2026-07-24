#pragma once

#include "../../io/IceCube/ICParameter.h"  // params::ic::nBarrParams
#include "../../io/IceCube/ICSample.h"

#include <cstddef>

namespace ana::ic {

  /**
   * Metal (Apple GPU) backend for the conventional + prompt atmospheric flux
   * histogram. Same design as ICMetalPowerlaw: a pure-C++ facade over an
   * Obj-C++/Metal implementation (ICMetalAtmo.mm), with a non-Apple stub
   * (ICMetalAtmo_stub.cpp) exposing available()==false.
   *
   * Per-event columns (e_true, conv/prompt baseline+alt, 4 conventional Barr
   * gradients, CSR bin_offsets) are copied to FP32 GPU buffers ONCE at
   * construction; each recalculate() only pushes the scalars that change.
   * FP32 weights + threadgroup tree reduction, matching the FP64 CPU path.
   */
  class ICMetalAtmo {
   public:
    ICMetalAtmo(const io::ic::ICSample& sample,
                double                  conv_delta_gamma_e_ref,
                double                  prompt_delta_gamma_e_ref);
    ~ICMetalAtmo();

    ICMetalAtmo(const ICMetalAtmo&)            = delete;
    ICMetalAtmo& operator=(const ICMetalAtmo&) = delete;

    /**
     * Dispatch the atmo kernel. barr points to params::ic::nBarrParams values
     * (H, W, Y, Z). per-event weights are only written when fill_per_event.
     */
    void recalculate(double        cr,
                     double        delta_gamma,
                     double        conv_norm,
                     double        prompt_norm,
                     const double* barr,
                     bool          fill_per_event);

    [[nodiscard]] const float* histogram() const noexcept;         // nBins floats
    [[nodiscard]] const float* per_event_weight() const noexcept;  // size() floats
    [[nodiscard]] std::size_t  size() const noexcept;

    [[nodiscard]] static bool available() noexcept;

   private:
    void* m_State = nullptr;  // opaque MetalState* (Obj-C++), or nullptr in the stub
  };

}  // namespace ana::ic
