#include "InputOptions.h"

// STL includes
#include <chrono>
#include <iostream>

// boost includes
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace io {

  InputOptions::InputOptions(int argc, char** argv, experiment_options_t experiment_options)
    : m_Seed(std::chrono::system_clock::now().time_since_epoch().count())
    , m_Silent(false)
    , m_MultiThreadingCores(-1)
    , m_ExperimentOptions(std::move(experiment_options)) {
    std::string inputFile;
    try {

      namespace po = boost::program_options;
      namespace pt = boost::property_tree;

      const long current_time = std::chrono::system_clock::now().time_since_epoch().count();

      po::options_description generic_options("Options");

      generic_options.add_options()("help,h", "Print help message")
      ("config,c", po::value<std::string>(&m_ConfigFile)->default_value("config.json")->required(), "Set Config File")
      ("seed", po::value<long>(&m_Seed)->default_value(current_time), "Set seed for simulation")
      ("silent", po::bool_switch(&m_Silent), "Run fit in silence mode")
      ("multiThreading,m", po::value<int>(&m_MultiThreadingCores)->default_value(1), "Use multiple threads for fitting")
      ("tolerance", po::value<double>(&m_Tolerance)->default_value(0.05), "Set Fit tolerance")
      ;

      po::options_description cmdline_options;
      cmdline_options.add(generic_options);
      for (const auto& [name, option] : m_ExperimentOptions) {
        cmdline_options.add(option->options());
      }

      po::variables_map vm;
      store(po::parse_command_line(argc, argv, cmdline_options), vm);

      if (vm.count("help")) {
        std::stringstream ss;
        ss << "Basic Command Line Parameter App\n"
           << generic_options << std::endl;

        throw std::logic_error(ss.str());
      }

      notify(vm);

      namespace pt = boost::property_tree;

      if (!boost::filesystem::exists(m_ConfigFile)) {
        throw std::invalid_argument("Error: Config File " + m_ConfigFile + " not found");
      }

      std::cout << "Reading Config File: " << m_ConfigFile << '\n';
      pt::read_json(m_ConfigFile, m_ConfigTree);

      m_InputParameter = std::make_shared<InputParameter>(m_ConfigTree.get_child("Parameter"));

      const auto experiment = m_ConfigTree.get_optional<std::string>("Experiment");

      const auto registered_names = [this] {
        std::string names;
        for (const auto& [name, option] : m_ExperimentOptions) {
          names += (names.empty() ? "" : ", ") + name;
        }
        return names;
      };

      if (!experiment) {
        throw std::invalid_argument("Config file is missing the top-level \"Experiment\" key. Registered experiments: " + registered_names());
      }

      m_Experiment = *experiment;

      const auto it = m_ExperimentOptions.find(m_Experiment);
      if (it == m_ExperimentOptions.end()) {
        throw std::invalid_argument("Unknown experiment \"" + m_Experiment + "\". Registered experiments: " + registered_names());
      }

      std::cout << "Selected experiment: " << m_Experiment << '\n';
      it->second->read(vm, m_ConfigTree);

    } catch (std::exception& e) {
      throw;
    }
  }
}  // namespace io