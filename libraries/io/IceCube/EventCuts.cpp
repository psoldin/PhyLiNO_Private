#include "EventCuts.h"

#include <boost/property_tree/ptree.hpp>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace io::ic {

  std::vector<EventCut> parse_event_cuts(const boost::property_tree::ptree& node,
                                         const std::string&                 sample_name) {
    std::vector<EventCut> cuts;

    const auto cuts_node = node.get_child_optional("Cuts");
    if (!cuts_node)
      return cuts;

    for (const auto& [cut_name, cut_node] : *cuts_node) {
      const std::string where = "parse_samples: sample '" + sample_name + "' cut '" + cut_name + "' ";

      EventCut cut{.name = cut_name, .branch = cut_node.get<std::string>("Branch", "")};

      const auto min = cut_node.get_optional<double>("Min");
      const auto max = cut_node.get_optional<double>("Max");
      if (min) cut.min = *min;
      if (max) cut.max = *max;

      if (cut.branch.empty())
        throw std::runtime_error(where + "has no \"Branch\" to cut on");
      if (!min && !max)
        throw std::runtime_error(where + "has neither \"Min\" nor \"Max\"; it would keep every event");
      if (std::isnan(cut.min) || std::isnan(cut.max))
        throw std::runtime_error(where + "has a NaN bound, which no event can pass");
      if (!(cut.min < cut.max))
        throw std::runtime_error(where + "has Min >= Max; it would keep no event");

      cuts.push_back(std::move(cut));
    }
    return cuts;
  }

  std::string describe_event_cuts(const std::vector<EventCut>& cuts) {
    std::ostringstream out;
    bool               first = true;
    for (const EventCut& cut : cuts) {
      if (!first) out << " and ";
      first = false;

      const bool has_min = cut.min != -std::numeric_limits<double>::infinity();
      const bool has_max = cut.max != std::numeric_limits<double>::infinity();
      if (has_min) out << cut.min << " < ";
      out << cut.branch;
      if (has_max) out << " < " << cut.max;
    }
    return out.str();
  }

}  // namespace io::ic
