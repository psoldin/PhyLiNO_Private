#pragma once

namespace params::ic {

  /**
   * Flat fit-parameter layout for the single IC86 tracks-only sample, ordered to
   * match the Minuit2 index array and the top-level "Parameter" list in the config.
   *
   * Layout mirrors the NNMFit tracks-only configuration (Fig_Configuration.yaml,
   * excluded_components: muon, conventional_veto, prompt_veto):
   *
   *   Astro (Powerlaw):      AstroNorm, SpectralIndex (= gamma_astro)
   *   Conventional atmo:     ConvNorm, BarrH, BarrW, BarrY, BarrZ
   *   Prompt atmo:           PromptNorm
   *   Shared atmo nuisances: CRGrad, DeltaGamma  (one shared param each, applied
   *                          to both conv and prompt, matching the YAML anchors)
   *
   * The trailing block is scaffolding for components whose input files are not
   * present in the repo/parquet yet (gated by config flags, no-op until loaded):
   *   MuonNorm  -> muontemplate (Corsika muon template pickle)
   *   DOMEff, IceAbs, IceScat -> SnowStorm detector-gradient systematics pickle
   * Keep these Fixed in the config while their component is disabled.
   */
  enum General : int {
    AstroNorm    = 0,  // astrophysical flux normalization
    SpectralIndex,     // gamma_astro: astro power-law index (reweight anchored at reference index)

    ConvNorm,          // conventional atmospheric normalization
    PromptNorm,        // prompt atmospheric normalization

    BarrH,             // conventional Barr parameters (conv only, per NNMFit config)
    BarrW,
    BarrY,
    BarrZ,
    _last_of_Barr_,

    CRGrad = _last_of_Barr_,  // cosmic-ray model gradient (H4a <-> GST4), shared conv+prompt
    DeltaGamma,               // atmospheric spectral tilt, shared conv+prompt
    _last_of_Flux_,

    // --- scaffolded systematics (no-op unless the component's file + flag are set) ---
    MuonNorm = _last_of_Flux_,  // atmospheric muon template normalization
    DOMEff,                     // DOM efficiency (SnowStorm gradient)
    IceAbs,                     // bulk ice absorption (SnowStorm gradient)
    IceScat,                    // bulk ice scattering (SnowStorm gradient)
    _last_of_General_
  };

  inline constexpr int nBarrParams =
    static_cast<int>(_last_of_Barr_) - static_cast<int>(BarrH);  // = 4 (H, W, Y, Z)

  inline constexpr int nDetSysParams =
    static_cast<int>(_last_of_General_) - static_cast<int>(DOMEff);  // = 3 (DOMEff, IceAbs, IceScat)

  constexpr int number_of_general_parameters() noexcept {
    return static_cast<int>(_last_of_General_);
  }

  constexpr int number_of_parameters() noexcept {
    return number_of_general_parameters();
  }

  static_assert(nBarrParams == 4, "Expected 4 Barr parameters: H, W, Y, Z");
  static_assert(number_of_parameters() == 14,
    "10 flux/atmo params + MuonNorm + 3 detector params. Update config + this if the layout changes.");

}  // namespace params::ic
