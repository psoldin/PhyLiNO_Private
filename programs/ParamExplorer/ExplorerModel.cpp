#include "ExplorerModel.h"

#include "ICComponentBreakdown.h"
#include "IceCube/ICParameter.h"
#include "Marginalize.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <span>
#include <string_view>
#include <utility>

namespace explorer {

  namespace {

    /**
     * The components of the stack, in draw order.
     *
     * Two spellings of the atmospheric contribution, and the caller picks one:
     * the single summed entry, or its conventional and prompt halves. Never
     * both -- the halves sum to the summed entry, so stacking all three would
     * double-count it.
     *
     * The summed entry is the default because the split is not free.
     * AtmosphericFlux::breakdown() re-walks every MC event to produce it, which
     * measures at ~163 ms on the combined config against ~0 for every other
     * component here (they are cached spans). See the Latency section of the
     * design doc.
     *
     * "systematicsDelta" is absent from both lists: it is a signed correction to
     * the total rather than a component, and stacking a signed quantity means
     * nothing.
     */
    constexpr std::string_view kSummedStack[] = {
      "astro", "atmospheric", "template", "galactic",
    };

    constexpr std::string_view kSplitStack[] = {
      "astro", "atmospheric_conv", "atmospheric_prompt", "template", "galactic",
    };

    /**
     * The per-component histograms the stack draws.
     *
     * The split path is result::ic::component_breakdown() itself, so the split
     * curves are literally the ones the results writers emit. The summed path
     * cannot call it -- it computes the conv/prompt split unconditionally, which
     * is the whole cost being avoided -- so it reads the same accessors
     * component_breakdown() reads for the entries it does keep. The one thing
     * the two paths must agree on is what "atmospheric" contains, and both take
     * it from SampleLikelihood::atmospheric_histogram().
     */
    [[nodiscard]] std::vector<std::pair<std::string, std::vector<double>>>
    breakdown_for(const ana::ic::SampleLikelihood& sample, const ana::ParameterWrapper& parameter, bool split) {
      if (split)
        return result::ic::component_breakdown(sample, parameter);

      std::vector<std::pair<std::string, std::vector<double>>> components;
      components.reserve(4);
      components.emplace_back("astro", sample.in_analysis_bins(sample.astro_histogram()));
      components.emplace_back(sample.config().wants_veto() ? "atmospheric_veto" : "atmospheric",
                              sample.in_analysis_bins(sample.atmospheric_histogram()));
      components.emplace_back("template", sample.in_analysis_bins(sample.template_histogram()));
      components.emplace_back("galactic", sample.in_analysis_bins(sample.galactic_histogram()));

      return components;
    }

  }  // namespace

  ExplorerModel::ExplorerModel(const std::string& config_path) {
    // io::Options parses a command line, so the config path is handed over the
    // way LLHFit does it -- through the ("config,c") option.
    std::string        program = "PhyLiNOExplorer";
    std::string        flag    = "-c";
    std::string        path    = config_path;
    std::array<char*, 3> argv{program.data(), flag.data(), path.data()};

    m_Module = std::make_shared<ana::ic::ICExperimentModule>();

    ana::module_map_t modules;
    modules[m_Module->name()] = m_Module;

    m_Options = std::make_shared<io::Options>(static_cast<int>(argv.size()), argv.data(),
                                              ana::collect_input_options(modules));

    const std::string& experiment = m_Options->inputOptions().experiment();
    if (experiment != m_Module->name())
      throw std::invalid_argument("The parameter explorer only knows the IceCube likelihood, but \"" +
                                  config_path + "\" selects experiment \"" + experiment + "\"");

    m_Llh = std::dynamic_pointer_cast<ana::ic::ICLikelihood>(m_Module->create_likelihood(m_Options, 0));
    if (!m_Llh)
      throw std::logic_error("ICExperimentModule did not return an ICLikelihood");

    const auto& input = m_Options->inputOptions().input_parameters();

    const int n = params::ic::number_of_parameters();
    m_Params.reserve(static_cast<std::size_t>(n));
    m_Info.reserve(static_cast<std::size_t>(n));

    for (int i = 0; i < n; ++i) {
      const double start = input.value(i);
      m_Params.push_back(start);

      const auto& lower = input.parameters()[i].lower_bound();
      const auto& upper = input.parameters()[i].upper_bound();

      const double step = input.uncertainty(i);

      // Bounds are optional and one-sided in practice: ConvNorm, PromptNorm and
      // MuonNorm declare a LowerBound only. A slider needs two finite ends, so
      // the missing side is invented.
      //
      // Generously. An earlier version used 2 * StartValue, which gave
      // PromptNorm a range of [0, 1] around a start of 0.5 -- too narrow to
      // push the component anywhere interesting, and the slider read as broken
      // because nothing visible happened. The span below reaches ~5x nominal
      // for the norms, which is the range AstroNorm declares explicitly
      // ([0, 5] around 1.77) and a reasonable read of what "explore this
      // parameter" means.
      const double reach = std::max({10.0 * step, std::abs(start), 1.0});
      const double lo    = lower ? *lower : start - reach;
      const double hi    = upper ? *upper : start + std::max(4.0 * std::abs(start - lo), reach);

      m_Info.push_back(ParamInfo{.name   = input.name(i),
                                 .index  = i,
                                 .value  = start,
                                 .start  = start,
                                 .step   = step,
                                 .asimov = input.parameters()[i].asimov_value(),
                                 .lo     = lo,
                                 .hi     = hi,
                                 .fixed  = input.fixed(i)});
    }

    m_Reference = evaluate();
  }

  void ExplorerModel::set(int index, double value) {
    m_Params.at(static_cast<std::size_t>(index)) = value;
    m_Info.at(static_cast<std::size_t>(index)).value = value;
    m_Stale = true;
  }

  void ExplorerModel::reset() {
    for (const ParamInfo& info : m_Info)
      set(info.index, info.start);
  }

  double ExplorerModel::evaluate() {
    const double llh = m_Llh->calculate_likelihood(m_Params.data());
    m_Stale = false;
    return llh;
  }

  std::vector<std::string> ExplorerModel::sample_names() const {
    std::vector<std::string> names;
    names.reserve(m_Llh->n_samples());
    for (std::size_t i = 0; i < m_Llh->n_samples(); ++i)
      names.push_back(m_Llh->sample(i).config().name);

    return names;
  }

  std::vector<AxisInfo> ExplorerModel::axes(std::size_t sample) const {
    const auto& binning = m_Llh->sample(sample).config().binning;

    std::vector<AxisInfo> result;
    result.reserve(binning.n_axes());
    for (const auto& axis : binning.axes())
      result.push_back(AxisInfo{.kind_name = std::string(io::ic::axis_kind_name(axis.kind)),
                                .edges     = axis_edges(axis)});

    return result;
  }

  Marginalized ExplorerModel::marginalize(std::size_t sample, std::size_t axis, bool split_atmospheric) {
    if (m_Stale)
      static_cast<void>(evaluate());

    const auto& likelihood = m_Llh->sample(sample);
    const auto& binning    = likelihood.config().binning;

    Marginalized result;
    result.edges = axis_edges(binning.axes()[axis]);
    result.total = project(likelihood.predicted(), binning, axis);
    result.data  = project(likelihood.data(), binning, axis);

    const auto breakdown = breakdown_for(likelihood, m_Llh->parameter(), split_atmospheric);

    const std::span<const std::string_view> wanted_keys =
      split_atmospheric ? std::span<const std::string_view>(kSplitStack)
                        : std::span<const std::string_view>(kSummedStack);

    for (const std::string_view wanted : wanted_keys) {
      // A veto-reweighted sample spells the summed entry "atmospheric_veto",
      // so the summed key matches either spelling.
      const auto it = std::find_if(breakdown.begin(), breakdown.end(), [wanted](const auto& entry) {
        return entry.first == wanted || (wanted == "atmospheric" && entry.first == "atmospheric_veto");
      });
      // A sample builds only the components its config declares, so an absent or
      // empty one is a normal configuration, not an error -- it is skipped
      // rather than drawn as a flat zero. A component that is present but
      // identically zero (a galactic norm fixed at a template the sample does
      // not carry) is skipped for the same reason: it would be a flat line
      // along the bottom of the plot and an entry in the legend.
      if (it == breakdown.end() || it->second.empty())
        continue;
      if (std::all_of(it->second.begin(), it->second.end(), [](double v) { return v == 0.0; }))
        continue;

      result.components.push_back(NamedHist{.name   = it->first,
                                            .values = project(it->second, binning, axis)});
    }

    return result;
  }

}  // namespace explorer
