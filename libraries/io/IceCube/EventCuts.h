#pragma once

#include <limits>
#include <string>
#include <vector>

#include <boost/property_tree/ptree_fwd.hpp>

namespace io::ic {

  /**
   * One per-event range cut on a continuous parquet column, declared in the
   * "Cuts" subtree of a "Samples.<name>" config node:
   *
   *   "Cuts": { "ants": { "Branch": "ANTSCORE", "Min": 0.9 } }
   *
   * Deliberately its own mechanism rather than a mode of the sample's Topology
   * cut: that one selects whole event classes by integer label, this one is a
   * threshold on a continuous score, and a sample may configure both at once
   * (ICDataBase intersects the two masks). Like the topology cut, these are
   * applied to the MC parquet and to a data parquet alike, before binning, so
   * both sides of the likelihood see one selection.
   *
   * Both bounds are exclusive and independently optional, so "Min" on its own
   * reads as `column > Min` -- the same comparison the pre-binned exports apply
   * (make_muon_template.ipynb, make_snowstorm_gradients.ipynb), which is what
   * makes a re-exported template comparable bin for bin.
   *
   * A NaN fails both comparisons and is dropped. That is the opposite of the
   * topology cut's NaN rule on purpose: a missing class label is still an event
   * of some class, but an event with no score was never scored, so no threshold
   * on that score can honestly keep it.
   */
  struct EventCut {
    std::string name;  // config key, for diagnostics
    std::string branch;
    double      min = -std::numeric_limits<double>::infinity();
    double      max = std::numeric_limits<double>::infinity();

    [[nodiscard]] bool keeps(double value) const noexcept { return value > min && value < max; }
  };

  /**
   * Parses the optional "Cuts" subtree of a "Samples.<name>" node; `sample_name`
   * only names the sample in error messages. Each child is one cut: a "Branch"
   * plus at least one of "Min" / "Max". A cut with no bound would keep every
   * event and a cut with Min >= Max would keep none, so both are config errors
   * rather than silent no-ops. Returns empty when there is no "Cuts" node.
   */
  [[nodiscard]] std::vector<EventCut> parse_event_cuts(const boost::property_tree::ptree& node,
                                                       const std::string&                 sample_name);

  /** The cuts as one human-readable expression ("ANTSCORE > 0.9"), for the load-time report. */
  [[nodiscard]] std::string describe_event_cuts(const std::vector<EventCut>& cuts);

}  // namespace io::ic
