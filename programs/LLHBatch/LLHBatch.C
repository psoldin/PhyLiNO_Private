/**
 * LLHBatch: many independent IceCube fits in one process, reading the MC once.
 *
 * A pseudo-experiment study runs thousands of fits that differ only in their data
 * counts, the muon template, the likelihood and which parameters are fixed where.
 * Run one LLHFit per fit and each spends most of its time re-reading the MC parquet
 * (on the ANTS sample: about 15 s of loading for a 2 s fit on a GPU). LLHBatch loads
 * the MC -- and on CUDA, uploads its columns -- once, and runs every fit against it.
 *
 *   LLHBatch <jobs file> [--workers N] [--force] [-- <LLHFit options for every job>]
 *
 * The jobs file lists one job per line: a directory holding the job's config.json,
 * optionally followed by LLHFit options for that job alone (blank lines and lines
 * starting with '#' are skipped). Every job is the equivalent of
 *
 *   cd <dir> && LLHFit -c config.json --fitOnly <common options> <job options>
 *
 * and writes <dir>/Output.json in the same format, plus <dir>/wall.txt (seconds,
 * likelihood build and fit) and, if the job fails, <dir>/error.txt instead. A job
 * whose Output.json exists is skipped unless --force is given, so a stopped batch
 * resumes where it left off. Everything a job prints goes to the one stdout; pass
 * --silent among the common options to keep it readable.
 *
 * Sharing the MC is only valid when every job describes the same MC: the configs
 * may differ from the first job's only in the Parameter array, IceCube.Likelihood,
 * IceCube.AsimovMode, IceCube.UseData, and per sample Template.File and DataCounts.
 * Anything else -- another MC file, binning, cut, oscillation or gradient file,
 * backend -- is rejected before the first fit. Each job's data histogram is still
 * read from its own config (ICDataBase::with_data).
 *
 * --workers N runs N jobs at a time (default 1). With --gpuDevices among the common
 * options, worker i uses the device LLHFit's scan worker i would; with -m/--threads,
 * each fit's OpenMP team size is the same as in LLHFit. Exit status: 0 if every job
 * succeeded, 1 if any failed, 2 for a bad command line or incompatible configs.
 */
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "ExperimentModule.h"
#include "Fit.h"
#include "IceCube/ICModule.h"
#include "Options.h"
#include "write_results.h"

#include <TROOT.h>

namespace {

  namespace fs = std::filesystem;

  struct Job {
    fs::path                 dir;
    std::vector<std::string> args;  // LLHFit options for this job only
  };

  struct Settings {
    fs::path                 jobs_file;
    int                      workers = 1;
    bool                     force   = false;
    std::vector<std::string> common;  // LLHFit options for every job
  };

  [[noreturn]] void usage(const std::string& problem) {
    std::cerr << "LLHBatch: " << problem << "\n"
              << "usage: LLHBatch <jobs file> [--workers N] [--force] [-- <LLHFit options for every job>]\n";
    std::exit(2);
  }

  Settings parse_command_line(const int argc, char** argv) {
    Settings s;
    int      i = 1;
    for (; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--") {
        ++i;
        break;
      }
      if (arg == "--workers") {
        if (++i >= argc) usage("--workers needs a value");
        s.workers = std::atoi(argv[i]);
        if (s.workers < 1) usage("--workers must be at least 1");
      } else if (arg == "--force") {
        s.force = true;
      } else if (arg == "-h" || arg == "--help") {
        usage("help");
      } else if (s.jobs_file.empty() && arg.rfind("-", 0) != 0) {
        s.jobs_file = arg;
      } else {
        usage("unknown argument '" + arg + "' (LLHFit options go after --)");
      }
    }
    for (; i < argc; ++i) s.common.emplace_back(argv[i]);
    if (s.jobs_file.empty()) usage("no jobs file given");
    for (const std::string& a : s.common)
      if (a == "-c" || a == "--config" || a == "--fitOnly" || a.rfind("--scanMode", 0) == 0)
        usage("'" + a + "' is set by LLHBatch itself (every job is a --fitOnly fit of <dir>/config.json)");
    return s;
  }

  std::vector<Job> read_jobs(const fs::path& path) {
    std::ifstream in(path);
    if (!in) usage("cannot open jobs file '" + path.string() + "'");
    std::vector<Job> jobs;
    std::string      line;
    while (std::getline(in, line)) {
      std::istringstream words(line);
      std::string        dir;
      if (!(words >> dir) || dir.front() == '#') continue;
      Job job{.dir = fs::absolute(dir)};
      for (std::string a; words >> a;) job.args.push_back(a);
      jobs.push_back(std::move(job));
    }
    if (jobs.empty()) usage("jobs file '" + path.string() + "' lists no jobs");
    return jobs;
  }

  /** The part of a config that the shared MC depends on: everything but the per-job fields. */
  nlohmann::json shared_part(nlohmann::json cfg) {
    cfg.erase("Parameter");
    if (cfg.contains("IceCube")) {
      auto& ic = cfg["IceCube"];
      for (const char* key : {"Likelihood", "AsimovMode", "UseData"}) ic.erase(key);
      if (ic.contains("Samples"))
        for (auto& sample : ic["Samples"]) {
          sample.erase("DataCounts");
          if (sample.contains("Template")) sample["Template"].erase("File");
        }
    }
    return cfg;
  }

  /** Throws unless every job's config is an IceCube config that may share the first one's MC. */
  void check_compatible(const std::vector<Job>& jobs) {
    nlohmann::json reference;
    for (std::size_t k = 0; k < jobs.size(); ++k) {
      const fs::path config = jobs[k].dir / "config.json";
      std::ifstream  in(config);
      if (!in) throw std::runtime_error("cannot open " + config.string());
      const nlohmann::json cfg = nlohmann::json::parse(in);
      if (cfg.value("Experiment", "") != "IceCube")
        throw std::runtime_error(config.string() + ": LLHBatch runs IceCube configs only");
      nlohmann::json part = shared_part(cfg);
      if (k == 0) {
        reference = std::move(part);
        continue;
      }
      if (part != reference) {
        const nlohmann::json diff = nlohmann::json::diff(reference, part);
        throw std::runtime_error(config.string() + " differs from " + (jobs[0].dir / "config.json").string() +
                                 " outside the per-job fields (Parameter, IceCube.Likelihood/AsimovMode/UseData, "
                                 "Samples.*.Template.File/DataCounts), so the two cannot share one MC: " +
                                 diff.dump());
      }
    }
  }

  /**
   * Run one job. The module and options are the job's own -- parsing a config fills
   * the module's option object -- but the module shares `cache_holder`'s MC.
   * Returns the line to report.
   */
  std::string run_job(const Job& job, const Settings& s, const ana::ic::ICExperimentModule& cache_holder,
                      const int worker_index, std::mutex& parse_mutex) {
    const auto t0     = std::chrono::steady_clock::now();
    auto       module = std::make_shared<ana::ic::ICExperimentModule>();
    module->share_cache_with(cache_holder);
    const ana::module_map_t modules{{module->name(), module}};

    std::vector<std::string> args{"LLHBatch", "-c", (job.dir / "config.json").string(), "--fitOnly"};
    args.insert(args.end(), s.common.begin(), s.common.end());
    args.insert(args.end(), job.args.begin(), job.args.end());
    std::vector<char*> argv;
    for (std::string& a : args) argv.push_back(a.data());

    std::shared_ptr<io::Options> options;
    {
      // boost::program_options and property_tree are not documented as
      // thread-safe; parsing is cheap next to a fit, so it is serialized.
      const std::scoped_lock lock(parse_mutex);
      options = std::make_shared<io::Options>(static_cast<int>(argv.size()), argv.data(),
                                              ana::collect_input_options(modules));
    }

    ana::Fit fit(options, module, worker_index);
    fit.minimize();
    result::write_results(fit, (job.dir / "Output").string());

    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::ofstream(job.dir / "wall.txt") << wall << '\n';
    std::ostringstream report;
    report << "converged " << (fit.converged() ? "yes" : "no") << ", " << wall << " s";
    return report.str();
  }

}  // namespace

int main(int argc, char** argv) {
  ROOT::EnableThreadSafety();
  const Settings         s    = parse_command_line(argc, argv);
  const std::vector<Job> jobs = read_jobs(s.jobs_file);

  try {
    check_compatible(jobs);
  } catch (const std::exception& e) {
    std::cerr << "LLHBatch: " << e.what() << '\n';
    return 2;
  }

  std::vector<const Job*> todo;
  for (const Job& job : jobs)
    if (s.force || !fs::exists(job.dir / "Output.json")) todo.push_back(&job);
  std::cout << "LLHBatch: " << jobs.size() << " jobs, " << (jobs.size() - todo.size())
            << " already done, " << todo.size() << " to run on " << std::min<std::size_t>(s.workers, todo.size())
            << " worker(s)\n";

  // Never fits; it only owns the cache every job's module shares. The first job
  // to build a likelihood loads the MC into it, the others wait on its lock.
  const ana::ic::ICExperimentModule cache_holder;
  std::mutex                        parse_mutex, print_mutex;
  std::atomic<std::size_t>          next{0}, done{0}, failed{0};
  const auto                        start = std::chrono::steady_clock::now();

  auto worker = [&](const int worker_index) {
    for (std::size_t k = next.fetch_add(1); k < todo.size(); k = next.fetch_add(1)) {
      const Job&  job = *todo[k];
      std::string report;
      bool        ok = true;
      try {
        fs::remove(job.dir / "error.txt");
        report = run_job(job, s, cache_holder, worker_index, parse_mutex);
      } catch (const std::exception& e) {
        ok     = false;
        report = std::string("FAILED: ") + e.what();
        std::ofstream(job.dir / "error.txt") << e.what() << '\n';
      }
      const std::size_t n       = ++done;
      const double      elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      if (!ok) ++failed;
      const std::scoped_lock lock(print_mutex);
      std::cout << "LLHBatch [" << n << "/" << todo.size() << ", " << static_cast<int>(elapsed) << " s] "
                << job.dir.string() << ": " << report << std::endl;
    }
  };

  std::vector<std::thread> threads;
  const int                n_workers = static_cast<int>(std::min<std::size_t>(s.workers, std::max<std::size_t>(todo.size(), 1)));
  for (int w = 0; w < n_workers; ++w) threads.emplace_back(worker, w);
  for (auto& t : threads) t.join();

  std::cout << "LLHBatch: " << (todo.size() - failed) << " of " << todo.size() << " jobs succeeded" << std::endl;
  return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
