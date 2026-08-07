#pragma once

#include "IceCube/ICLikelihood.h"
#include "IceCube/SampleLikelihood.h"
#include "ParameterWrapper.h"

// STL includes
#include <string>
#include <utility>
#include <vector>

namespace result::ic {

  /**
   * Per-component breakdown of one sample's prediction, both summed and per-bin:
   * a mis-scaled template or gradient is visible here instead of hidden inside
   * the sample total. Every component is reported in the sample's ANALYSIS
   * binning -- the per-event and 2D-template components are spread over the RA
   * axis exactly as the prediction is -- so an external per-bin diff
   * (tools/nnmfit_oracle/compare_to_nnmfit.py) needs no reshaping.
   *
   * Keys, in the order returned:
   *   "astro"
   *   "atmospheric" | "atmospheric_veto"   (whichever variant the sample declared,
   *                                         so a diff can tell a veto-reweighted
   *                                         sample from a plain one)
   *   "template"
   *   "systematicsDelta"
   *   "galactic"
   *   "atmospheric_conv"                   conventional half of the entry above,
   *   "atmospheric_prompt"                 and its prompt half. The two sum to the
   *                                        "atmospheric"/"atmospheric_veto" entry
   *                                        (exactly on the CPU path; to FP32
   *                                        precision when the fit ran on an FP32
   *                                        GPU backend, since the split is always
   *                                        recomputed in FP64 on the CPU). Both
   *                                        carry the veto passing fraction when
   *                                        the sample uses it, and both are empty
   *                                        when the sample has no atmospheric
   *                                        component.
   *
   * `parameter` must be the point the sample's prediction currently holds -- the
   * writers re-evaluate the likelihood at the minimum before calling this.
   *
   * Both output formats (JSON and protobuf) go through this one function so they
   * cannot disagree about what a component is called or contains.
   */
  inline std::vector<std::pair<std::string, std::vector<double>>>
  component_breakdown(const ana::ic::SampleLikelihood& sample, const ana::ParameterWrapper& parameter) {
    const std::string atmo_key = sample.config().wants_veto() ? "atmospheric_veto" : "atmospheric";

    const ana::ic::AtmoBreakdown atmo_split = sample.atmospheric_breakdown(parameter);

    std::vector<std::pair<std::string, std::vector<double>>> components;
    components.reserve(7);
    components.emplace_back("astro", sample.in_analysis_bins(sample.astro_histogram()));
    components.emplace_back(atmo_key, sample.in_analysis_bins(sample.atmospheric_histogram()));
    components.emplace_back("template", sample.in_analysis_bins(sample.template_histogram()));
    components.emplace_back("systematicsDelta", sample.in_analysis_bins(sample.systematics_mu_delta()));
    components.emplace_back("galactic", sample.in_analysis_bins(sample.galactic_histogram()));
    components.emplace_back("atmospheric_conv", sample.in_analysis_bins(atmo_split.conv));
    components.emplace_back("atmospheric_prompt", sample.in_analysis_bins(atmo_split.prompt));

    return components;
  }

  inline double sum_of(const std::vector<double>& values) {
    double total = 0.0;
    for (const double v : values)
      total += v;
    return total;
  }

  /**
   * Put the likelihood back at the fitted minimum before its components are read.
   * The minimizer's last call is not guaranteed to have landed on the minimum, so
   * every writer does this once, for the prediction as much as for the breakdown.
   */
  inline void evaluate_at_minimum(ana::ic::ICLikelihood& llh, const double* minimum) {
    [[maybe_unused]] const double llh_at_minimum = llh.calculate_likelihood(minimum);
  }

}  // namespace result::ic
