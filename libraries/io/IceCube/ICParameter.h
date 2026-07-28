#pragma once

namespace params::ic {

  /**
   * Flat fit-parameter layout, ordered to match the Minuit2 index array and the
   * top-level "Parameter" list in the config. One layout is shared by every
   * analysis sample: the flux parameters below are global (same astro/atmo
   * across samples, matching the NNMFit YAML anchors), so enabling or disabling
   * a sample does not change the layout.
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
   * The trailing block covers the per-sample components (a sample builds only what
   * its config declares, so these are Fixed in a config whose samples do not use
   * them):
   *   MuonNorm     -> muontemplate, the Corsika muon template (tracks)
   *   MuonGunNorm  -> muon, the MuonGun template (cascade samples)
   *   VetoThreshold-> effective_veto, shared by every veto-reweighted sample
   *   DOMEff, IceAbs, IceScat, HoleIceP0, HoleIceP1 -> SnowStorm detector gradients
   *     (shared parameter names; the gradient file itself is per sample)
   *   GalacticNorm0, GalacticNorm1 -> the galactic-plane templates a sample declares
   *     ("Galactic" subtree); NNMFit names these per model (cringefits_norm,
   *     unresolved_norm, fermi_norm, ...). Fixed in a config with no galactic component.
   *   AstroGamma1, AstroGamma2, AstroEBreak -> NNMFit's AstroBPL broken power law,
   *     active only when IceCube.AstroModel is "BrokenPowerlaw" (they replace
   *     SpectralIndex there). Fixed in a single-power-law config.
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

    // --- muon templates: one norm per template kind, a sample declares at most one ---
    MuonNorm = _last_of_Flux_,  // Corsika muon template normalization (tracks)
    MuonGunNorm,                // MuonGun template normalization (cascade samples)

    // --- veto (NNMFit effective_veto): shared by every veto-reweighted sample ---
    VetoThreshold,

    // --- SnowStorm detector gradients, in the order the exported gradient file uses ---
    DOMEff,     // DOM efficiency
    IceAbs,     // bulk ice absorption
    IceScat,    // bulk ice scattering
    HoleIceP0,  // hole-ice forward p0
    HoleIceP1,  // hole-ice forward p1
    _last_of_DetSys_,

    // --- galactic plane (NNMFit GalacticTemplate): one norm per declared template,
    // in the sample's config order. Global like every other flux parameter, so the
    // same model scales identically across samples. ---
    GalacticNorm0 = _last_of_DetSys_,
    GalacticNorm1,

    // --- astrophysical broken power law (NNMFit AstroBPL), used when
    // IceCube.AstroModel is "BrokenPowerlaw". AstroNorm is the shared norm; in
    // that mode SpectralIndex is unused and the three below take over. Fixed in a
    // single-power-law config. AstroEBreak is log10(E_break / GeV). ---
    AstroGamma1,
    AstroGamma2,
    AstroEBreak,
    _last_of_General_
  };

  inline constexpr int nBarrParams =
    static_cast<int>(_last_of_Barr_) - static_cast<int>(BarrH);  // = 4 (H, W, Y, Z)

  inline constexpr int nDetSysParams =
    static_cast<int>(_last_of_DetSys_) - static_cast<int>(DOMEff);  // = 5

  constexpr int number_of_general_parameters() noexcept {
    return static_cast<int>(_last_of_General_);
  }

  constexpr int number_of_parameters() noexcept {
    return number_of_general_parameters();
  }

  static_assert(nBarrParams == 4, "Expected 4 Barr parameters: H, W, Y, Z");
  static_assert(nDetSysParams == 5,
    "DOMEff, IceAbs, IceScat, HoleIceP0, HoleIceP1 -- the order the exported gradient file uses");
  static_assert(number_of_parameters() == 23,
    "10 flux/atmo params + 2 template norms + VetoThreshold + 5 detector params + 2 galactic norms "
    "+ 3 astro broken-power-law params. "
    "Update every config's Parameter array and this if the layout changes.");

}  // namespace params::ic
