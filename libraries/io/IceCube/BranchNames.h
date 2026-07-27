#pragma once

#include <array>
#include <string>

#include "ICParameter.h"  // params::ic::nBarrParams

namespace io::ic {

  /**
   * Names of the parquet columns read into ICSample. Defaults match the
   * tracks-only baseline dataset (dataset_tracks_baseline.parquet); override
   * any of them via the "IceCube.Branches" config subtree if the schema differs,
   * or per-sample via the "Branches" subtree of an "IceCube.Samples" entry.
   */
  struct BranchNames {
    std::string reco_energy     = "energy_truncated";
    std::string reco_zenith     = "zenith_MPEFit";
    std::string true_energy     = "MCPrimaryEnergy";
    std::string astro_baseline  = "powerlaw";
    std::string conv_baseline   = "mceq_conv_H4a_SIBYLL23c";
    std::string conv_alt        = "mceq_conv_GST4_SIBYLL23c";
    std::string prompt_baseline = "mceq_pr_H4a_SIBYLL23c";
    std::string prompt_alt      = "mceq_pr_GST4_SIBYLL23c";
    // Conventional Barr gradients, order matches params::ic {BarrH, BarrW, BarrY, BarrZ}.
    std::array<std::string, params::ic::nBarrParams> barr_conv = {
        "barr_h_mceq_H4a_SIBYLL23c",
        "barr_w_mceq_H4a_SIBYLL23c",
        "barr_y_mceq_H4a_SIBYLL23c",
        "barr_z_mceq_H4a_SIBYLL23c"};

    // Veto passing-fraction coefficients {a, b, c} of NNMFit's second-order
    // log10(PF) expansion, one set per atmospheric component. Only read for
    // samples that declare the veto components -- the tracks baseline parquet
    // does not carry these columns.
    std::array<std::string, 3> veto_conv = {
        "log_PF_at100GeV_conv_a",
        "log_PF_at100GeV_conv_b",
        "log_PF_at100GeV_conv_c"};
    std::array<std::string, 3> veto_prompt = {
        "log_PF_at100GeV_pr_a",
        "log_PF_at100GeV_pr_b",
        "log_PF_at100GeV_pr_c"};
  };

}  // namespace io::ic
