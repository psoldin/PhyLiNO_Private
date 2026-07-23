#pragma once

namespace params::ic {

  enum Parameter : int {
    AstroNorm = 0,
    SpectralIndex,

    ConvNorm,
    PromptNorm,

    BarrH,
    BarrW,
    BarrY,
    BarrZ,

    CRGrad,
    DeltaGamma,

    MuonNorm,
    DOMEff,
    IceAbs,
    IceScat,
    _last_of_parameter_
  };

  constexpr int number_of_parameters() noexcept {
    return static_cast<int>(_last_of_parameter_);
  }

}