#include "ScanCommon.h"

// STL includes
#include <array>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>

// includes
#include "IceCube/ICBlinding.h"
#include "IceCube/ICWriteResultsProto.h"

namespace llhfit {

  namespace {

    /// seedquality::start_vector() plus the one-off report of a partial block.
    std::vector<double> start_vector(const io::InputParameter& input_parameters, const std::map<std::string, double>& stored) {
      const auto& names = input_parameters.names();

      std::size_t filled = 0;
      const auto  values = seedquality::start_vector(
          names, stored, [&](std::size_t i) { return input_parameters.value(static_cast<int>(i)); }, filled);

      // Said once per run, not once per scan point: under --blind every file is
      // partial and the message would otherwise be the bulk of the output.
      if (!values.empty() && filled < names.size()) {
        static bool reported = false;
        if (!reported) {
          reported = true;
          std::cout << "Stored fits carry " << filled << " of " << names.size()
                    << " parameters; the rest start from their configured values (--blind drops the signal parameters)\n";
        }
      }

      return values;
    }

    /// A result file as it was written: its fit summary and every parameter it
    /// names, keyed by name, before anything decides what the values are good for.
    struct StoredResult {
      bool                          converged = false;
      double                        edm       = std::numeric_limits<double>::infinity();
      double                        llh       = 0.0;
      std::map<std::string, double> values;
      std::map<std::string, double> errors;
    };

    /**
     * @brief Reads a result file in one output format.
     *
     * @param name   Base name of the file, without its extension.
     * @param format "json" or "protobuf".
     * @return The result, or nullopt if the file is absent or holds no usable likelihood.
     */
    std::optional<StoredResult> read_stored_result(const std::string& name, const std::string& format) {
      StoredResult result;

      if (format == "protobuf") {
        const auto stored = result::ic::read_ice_cube_results_protobuf(name);
        if (!stored)
          return std::nullopt;

        result.converged = stored->converged;
        result.edm       = stored->edm;
        result.llh       = stored->llh;
        result.values    = stored->parameter_values;
        result.errors    = stored->parameter_errors;
      } else {
        const std::filesystem::path path = name + ".json";
        if (!std::filesystem::exists(path))
          return std::nullopt;

        std::ifstream  file(path);
        nlohmann::json stored = nlohmann::json::parse(file, nullptr, false);
        if (stored.is_discarded() || !stored.contains("LLH"))
          return std::nullopt;

        result.llh = stored["LLH"].get<double>();
        if (!std::isfinite(result.llh))
          return std::nullopt;

        result.converged = stored.value("converged", false);

        // A result written before EDM was recorded leaves this at infinity, which
        // falls the seed test back onto the converged flag alone.
        result.edm = stored.value("EDM", std::numeric_limits<double>::infinity());

        if (stored.contains("parameters")) {
          for (const auto& [key, parameter] : stored["parameters"].items()) {
            if (parameter.contains("value"))
              result.values[key] = parameter["value"].get<double>();
            if (parameter.contains("error") && parameter["error"].is_number())
              result.errors[key] = parameter["error"].get<double>();
          }
        }
      }

      return result;
    }

    /// The extensions the result writers append, longest first so ".pb.gz" is
    /// matched before a bare ".gz" ever could be.
    constexpr std::array kResultExtensions{".pb.gz", ".json"};

    /// Base name of a result, whether it was given with or without its extension.
    /// stored_fit() appends the extension itself, so "Output.json" would otherwise
    /// send it looking for "Output.json.json".
    std::string result_base_name(std::string path) {
      for (const std::string_view extension : kResultExtensions) {
        if (path.size() > extension.size() && path.ends_with(extension)) {
          path.resize(path.size() - extension.size());
          break;
        }
      }
      return path;
    }

    std::vector<double> load_seed_from_values(const std::shared_ptr<io::Options>& options) {
      const auto& input = options->inputOptions();
      if (input.seed_from().empty())
        return {};

      if (input.randomize_seeds()) {
        std::cout << "--seedFrom ignored: --randomizeSeeds sets the start values instead\n";
        return {};
      }

      const std::string path = result_base_name(input.seed_from());

      const auto& input_parameters = input.input_parameters();
      const auto& format           = input.output_format();

      // Both formats are tried whichever way round: a scan writing protobuf can be
      // seeded from the JSON fit that found the minimum, and the reverse.
      auto stored = stored_fit(path, input_parameters, format);
      if (!stored)
        stored = stored_fit(path, input_parameters, format == "protobuf" ? "json" : "protobuf");

      if (!stored)
        throw std::runtime_error("--seedFrom \"" + input.seed_from() + "\": no readable result there (looked for " + path + ".json and " + path + ".pb.gz)");

      if (stored->parameters.empty())
        throw std::runtime_error("--seedFrom \"" + input.seed_from() + "\": the result holds no usable start values -- either it stopped too far from its minimum to seed anything (see SeedQuality.h) or its parameter names do not match this config");

      std::cout << "Start values from " << path << " (likelihood " << stored->llh << ")\n";
      return stored->parameters;
    }

  }  // namespace

  std::optional<StoredFit> stored_fit(const std::string& name, const io::InputParameter& input_parameters, const std::string& format) {
    const auto stored = read_stored_result(name, format);
    if (!stored)
      return std::nullopt;

    const auto& [converged, edm, llh, values, errors] = *stored;

    StoredFit result;
    result.llh = llh;

    // A fit seeds its neighbours when it got close enough to its own minimum,
    // which EDM measures and Migrad's converged flag does not. See SeedQuality.h.
    if (seedquality::usable(converged, edm)) {
      result.parameters = start_vector(input_parameters, values);
      if (!result.parameters.empty()) {
        const auto& names = input_parameters.names();
        result.errors.assign(names.size(), 0.0);
        for (std::size_t i = 0; i < names.size(); ++i)
          if (const auto it = errors.find(names[i]); it != errors.end() && std::isfinite(it->second))
            result.errors[i] = it->second;
      }
    }

    return result;
  }

  void apply_start_values(ROOT::Math::Minimizer& minimizer, const io::InputParameter& input_parameters, const std::vector<double>& values) {
    const auto& parameters = input_parameters.parameters();
    if (values.size() != parameters.size())
      return;

    for (std::size_t i = 0; i < parameters.size(); ++i) {
      if (minimizer.IsFixedVariable(static_cast<unsigned int>(i)))
        continue;

      double start = values[i];

      // Minuit2's internal transformation of a bounded parameter is singular on
      // the limit itself, and a converged fit can come arbitrarily close to one.
      // Same margin the configured start values are pulled back by in Fit.cpp.
      const std::optional<double>& lower = parameters[i].lower_bound();
      const std::optional<double>& upper = parameters[i].upper_bound();
      if (lower || upper) {
        const double margin = 1.0e-3 * parameters[i].uncertainty();
        if (lower)
          start = std::max(start, *lower + margin);
        if (upper)
          start = std::min(start, *upper - margin);
      }

      minimizer.SetVariableValue(static_cast<unsigned int>(i), start);
    }
  }

  void apply_extra_fixed(ROOT::Math::Minimizer& min, const FixedParameters& fixed) {
    for (const auto& [index, value] : fixed) {
      min.SetVariableValue(static_cast<unsigned int>(index), value);
      min.FixVariable(static_cast<unsigned int>(index));
    }
  }

  std::string node_name(const scan::Node& node) {
    std::stringstream name;
    name << "Output_" << node.x << '_' << node.y;
    return name.str();
  }

  std::string node_name(const std::string& parameter, int node) {
    std::stringstream name;
    name << "Output_" << parameter << '_' << node;
    return name.str();
  }

  /// Distinct from node_name(const scan::Node&) above: a regular-grid node
  /// (i, j) sits at a different physical point than an adaptive-grid node of
  /// the same integer coordinates, so the two must never share a resume file.
  std::string node_name_regular(const scan::Node& node) {
    std::stringstream name;
    name << "OutputRegular_" << node.x << '_' << node.y;
    return name.str();
  }

  /// 1D counterpart of node_name_regular(const scan::Node&) above -- distinct
  /// from node_name(parameter, node) so a regular-grid point never shares a
  /// resume file with an adaptive-grid point at the same integer index.
  std::string node_name_regular(const std::string& parameter, int node) {
    std::stringstream name;
    name << "OutputRegular_" << parameter << '_' << node;
    return name.str();
  }

  bool resolve_warm_start(const std::shared_ptr<io::Options>& options) {
    const auto& input      = options->inputOptions();
    const bool  warm_start = input.scan_warm_start() && !input.randomize_seeds();
    if (input.scan_warm_start() && !warm_start)
      std::cout << "Warm start off: --randomizeSeeds already sets the start point of every scan fit\n";
    return warm_start;
  }

  const std::vector<double>& seed_from_values(const std::shared_ptr<io::Options>& options) {
    static const std::vector<double> values = load_seed_from_values(options);
    return values;
  }

  const std::vector<double>& evaluate_at_values(const std::shared_ptr<io::Options>& options) {
    static const std::vector<double> values = [&]() -> std::vector<double> {
      const auto& input = options->inputOptions();
      if (input.evaluate_at().empty())
        return {};

      const std::string path   = result_base_name(input.evaluate_at());
      const auto&       format = input.output_format();

      auto stored = read_stored_result(path, format);
      if (!stored)
        stored = read_stored_result(path, format == "protobuf" ? "json" : "protobuf");
      if (!stored)
        throw std::runtime_error("--evaluateAt \"" + input.evaluate_at() + "\": no readable result there (looked for " + path + ".json and " + path + ".pb.gz)");

      std::cout << "Evaluating at the parameters of " << path << " (likelihood " << stored->llh << " there)\n";
      if (!seedquality::usable(stored->converged, stored->edm))
        std::cout << "Warning: that result did not converge (EDM " << stored->edm << "); evaluating at its parameters anyway\n";

      const auto&         input_parameters = input.input_parameters();
      const auto&         names            = input_parameters.names();
      std::vector<double> point(names.size());
      std::string         missing;
      for (std::size_t i = 0; i < names.size(); ++i) {
        if (const auto it = stored->values.find(names[i]); it != stored->values.end()) {
          point[i] = it->second;
        } else {
          point[i] = input_parameters.value(static_cast<int>(i));
          missing += (missing.empty() ? "" : ", ") + names[i];
        }
      }
      if (!missing.empty())
        std::cout << "Not in that result, kept at their configured StartValue: " << missing << '\n';

      std::string unknown;
      for (const auto& [name, value] : stored->values)
        if (std::ranges::find(names, name) == names.end())
          unknown += (unknown.empty() ? "" : ", ") + name;
      if (!unknown.empty())
        std::cout << "In that result but not in this config, ignored: " << unknown << '\n';

      return point;
    }();
    return values;
  }

  void write_or_check_grid(const std::string& path, const nlohmann::json& grid, const std::string& kind, const std::string& file_pattern) {
    if (std::filesystem::exists(path)) {
      std::ifstream  existing_file(path);
      nlohmann::json existing = nlohmann::json::parse(existing_file);
      if (existing != grid)
        throw std::runtime_error(path + " in this directory describes a different " + kind + "; the " + file_pattern + " files here belong to that one. Scan into an empty directory instead.");
    } else {
      std::ofstream(path) << grid.dump(2) << '\n';
    }
  }

  std::vector<std::string> free_parameter_names(const std::shared_ptr<io::Options>& options) {
    const auto& input_parameters = options->inputOptions().input_parameters();
    const auto& names            = input_parameters.names();
    const bool  blind            = options->inputOptions().blind();

    std::vector<std::string> result;
    for (std::size_t i = 0; i < input_parameters.size(); ++i)
      if (!input_parameters.fixed(static_cast<int>(i)) && !(blind && result::ic::is_blinded_parameter(names[i])))
        result.push_back(names[i]);
    return result;
  }

  int parameter_index(const io::InputParameter& input_parameters, const std::string& parameter_name) {
    const auto& names = input_parameters.names();

    const auto found = std::ranges::find(names, parameter_name);
    if (found == names.end())
      throw std::runtime_error("No parameter named '" + parameter_name + "' in the config");

    return static_cast<int>(std::ranges::distance(names.begin(), found));
  }

  std::string result_extension(const std::string& format) {
    return format == "protobuf" ? ".pb.gz" : ".json";
  }

  void reuse_sibling_free_fit(const std::string& prefix, const std::string& label, const std::vector<std::string>& siblings, const io::InputParameter& input_parameters, const std::string& format) {
    const std::string            extension = result_extension(format);
    const std::filesystem::path  target    = prefix + label + extension;
    if (std::filesystem::exists(target))
      return;

    for (const auto& sibling : siblings) {
      if (sibling == label)
        continue;
      const std::string source = prefix + sibling;
      const auto        stored = stored_fit(source, input_parameters, format);
      if (!stored || stored->parameters.empty())
        continue;

      std::filesystem::copy_file(source + extension, target);
      std::cout << "Reusing the free fit of " << sibling << " for " << label << '\n';
      return;
    }
  }

}  // namespace llhfit
