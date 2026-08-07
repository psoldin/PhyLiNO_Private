#pragma once

#include "Fit.h"

#include "IceCube/ICInputOptions.h"
#include "IceCube/ICLikelihood.h"

// STL includes
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

}  // namespace result::ic
