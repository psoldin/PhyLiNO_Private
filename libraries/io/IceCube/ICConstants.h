#pragma once

namespace io::ic::Constants {

  // Legacy fixed tracks grid. Analysis binning is runtime and per-sample now
  // (see Binning.h / SampleConfig.h); what is left here only sizes the
  // scaffolded per-bin components (MuonTemplate, DetectorSystematics), which
  // still assume the tracks grid and are disabled until they are made
  // per-sample along with the cascade samples.
  //
  // The grid matches the NNMFit tracks-only config
  // (IC86_pass2_SnowStorm_v2_tracks_2D_binning):
  //   reco_energy: (2.5, 7, 46, log)  -> 46 edges in log10(E/GeV) => 45 bins
  //   reco_zenith: (-1, 0.0872, 34, cos) -> 34 edges in cos(zenith) => 33 bins
  // NNMFit's binning string convention is (min, max, N_edges, type); N is the
  // number of edges, so the bin count is N-1 in each dimension.

  // Reconstructed energy axis: uniform in log10(E/GeV), 45 bins from 2.5 to 7.0.
  inline constexpr int    nEnergyBins    = 45;
  inline constexpr double LogEnergyMin   = 2.5;
  inline constexpr double LogEnergyMax   = 7.0;
  inline constexpr double LogEnergyStep  = (LogEnergyMax - LogEnergyMin) / nEnergyBins;

  // Reconstructed zenith axis: uniform in cos(zenith), 33 bins from -1 to 0.0872
  // (cos(zenith) in [-1, 0.0872] => zenith from 180 deg down to ~85 deg).
  inline constexpr int    nZenithBins    = 33;
  inline constexpr double CosZenithMin   = -1.0;
  inline constexpr double CosZenithMax   = 0.0872;
  inline constexpr double CosZenithStep  = (CosZenithMax - CosZenithMin) / nZenithBins;

  // Total flattened bins (energy x zenith, row-major: E is the outer index).
  inline constexpr int nBins = nEnergyBins * nZenithBins;

}  // namespace io::ic::Constants
