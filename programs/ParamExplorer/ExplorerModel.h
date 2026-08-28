#pragma once

#include "IceCube/ICLikelihood.h"
#include "IceCube/ICModule.h"
#include "Options.h"

#include <memory>
#include <string>
#include <vector>

namespace explorer {

  /** One fit parameter as the GUI needs it: identity, current value, slider range. */
  struct ParamInfo {
    std::string name;
    int         index = 0;
    double      value = 0.0;
    // The point the Asimov data was generated at ("AsimovValue", defaulting to
    // the start value). Not the same as the start value in general: a config can
    // deliberately seed the fit off truth, and config_icecube_combined.json does
    // exactly that for SpectralIndex.
    double      asimov = 0.0;
    // Slider range. Real config bounds where the config gives them, the
    // fallback of the design doc where it does not -- several norms declare a
    // LowerBound only.
    double      lo    = 0.0;
    double      hi    = 0.0;
    bool        fixed = false;
  };

  /** One analysis axis of a sample, named for the axis combo box. */
  struct AxisInfo {
    std::string         kind_name;
    std::vector<double> edges;
  };

  /** One curve of the stack: a component name and its marginalized bins. */
  struct NamedHist {
    std::string         name;
    std::vector<double> values;
  };

  /** A sample's marginalized prediction along one axis, plus the data to overlay. */
  struct Marginalized {
    std::vector<double>    edges;
    std::vector<NamedHist> components;
    std::vector<double>    data;
  };

  /**
   * The IceCube forward model behind the parameter explorer, with no Qt in it.
   *
   * Loads the parquet baseline once (via ICExperimentModule, which caches the
   * ICDataBase and the GPU backend) and then answers two questions repeatedly:
   * what is -2lnL at the current parameter point, and what does the prediction
   * look like projected onto one axis of one sample.
   *
   * No minimizer runs: set() moves one parameter and nothing profiles over the
   * others. This is a viewer of the forward model, not a fit.
   */
  class ExplorerModel {
   public:
    /** Throws whatever the option parser and the parquet loaders throw. */
    explicit ExplorerModel(const std::string& config_path);

    [[nodiscard]] const std::vector<ParamInfo>& parameters() const noexcept { return m_Info; }

    /** Move one parameter. Cheap: writes the array, evaluates nothing. */
    void set(int index, double value);

    /** -2lnL at the current point. */
    [[nodiscard]] double evaluate();

    /** -2lnL at the config's start values, for the status bar's delta. */
    [[nodiscard]] double reference_likelihood() const noexcept { return m_Reference; }

    [[nodiscard]] std::vector<std::string> sample_names() const;
    [[nodiscard]] std::vector<AxisInfo>    axes(std::size_t sample) const;

    /**
     * The current prediction of one sample, projected onto one axis.
     *
     * Re-evaluates first when a set() has happened since the last evaluate():
     * the component histograms are whatever point the prediction currently
     * holds, so asking for them after moving a parameter but before evaluating
     * would silently return the previous point's stack.
     *
     * `split_atmospheric` draws the conventional and prompt halves as separate
     * curves instead of their sum. It is off by default because it is the
     * expensive option: it re-walks every MC event (~163 ms on the combined
     * config) where every other component is a cached span.
     */
    [[nodiscard]] Marginalized marginalize(std::size_t sample, std::size_t axis,
                                           bool split_atmospheric = false);

   private:
    // Declared in load order: the module owns the cached ICDataBase the
    // likelihood's samples read, so it must outlive the likelihood.
    std::shared_ptr<io::Options>                   m_Options;
    std::shared_ptr<ana::ic::ICExperimentModule>   m_Module;
    std::shared_ptr<ana::ic::ICLikelihood>         m_Llh;

    std::vector<double>    m_Params;
    std::vector<ParamInfo> m_Info;

    double m_Reference = 0.0;
    bool   m_Stale     = true;
  };

}  // namespace explorer
