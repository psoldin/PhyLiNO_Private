#pragma once

// The scans LLHFit can run, one per --scanMode. Each is documented where it is
// defined: Scan2D.cpp, Scan1D.cpp and ScanPairs.cpp.

// STL includes
#include <memory>
#include <string>

#include <nlohmann/json.hpp>

// includes
#include "ExperimentModule.h"
#include "Options.h"

namespace llhfit {

  // Scan2D.cpp

  void perform_2d_scan(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module);
  void perform_2d_scan_regular(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, int points_x = 30, int points_y = 30);

  // Scan1D.cpp
  void perform_1d_scan_regular(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const std::string& parameter_name, int points);
  void perform_1d_scan(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const std::string& parameter_name);
  void perform_nm1_scan(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const std::string& target);
  nlohmann::json perform_1d_crossing(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, const std::string& parameter_name, int index);
  void perform_1d_crossing_all(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module);
  void perform_1d_scan_all(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module, bool regular = false, int points = 30);

  // ScanPairs.cpp
  void perform_2d_stencil(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module);
  void perform_2d_pairs(std::shared_ptr<io::Options> options, std::shared_ptr<ana::ExperimentModule> module);

}  // namespace llhfit
