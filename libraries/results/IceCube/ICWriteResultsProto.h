#pragma once

#include "Fit.h"

#include "IceCube/ICInputOptions.h"
#include "IceCube/ICLikelihood.h"

// STL includes
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace result::ic {

  // Same content as get_json_file() in ICWriteResults.h, built into a
  // result::ic::FitResult protobuf message instead.
  //
  // Defined out-of-line (ICWriteResultsProto.cpp) so this header does not need
  // to include the generated ic_result.pb.h, keeping the generated protobuf
  // headers out of every translation unit that just wants to call
  // write_ice_cube_results_protobuf().
  void write_ice_cube_results_protobuf(ana::Fit& fit, ana::ic::ICLikelihood& llh, const io::ic::ICInputOptions& info, std::string_view name);

  /// The subset of a stored .pb.gz result a scan resume needs to skip a point
  /// or seed the next fit's start values. Kept free of the generated
  /// ic_result.pb.h for the same reason as the write side above.
  struct ProtoStoredFit {
    bool                          converged = false;
    double                        llh       = 0.0;
    std::map<std::string, double> parameter_values;  ///< Fitted value, keyed by config name.
  };

  /// Reads back a point an earlier run already fitted with --output-format
  /// protobuf. Returns nullopt if "<name>.pb.gz" is absent or unreadable.
  std::optional<ProtoStoredFit> read_ice_cube_results_protobuf(std::string_view name);

}  // namespace result::ic
