#include "ICModule.h"

#include "../Fit.h"
#include "CudaBackend.h"
#include "MetalBackend.h"

#include "IceCube/ICWriteResults.h"

#include <iostream>
#include <memory>
#include <stdexcept>

namespace ana::ic {

  namespace {

    // One GPU backend behind a given device's fit sessions; nullptr => CPU path.
    // A requested GPU backend falls back to CPU if no matching device is
    // present. device is only consulted for Cuda: Metal has no --gpuDevices
    // support (Apple builds run one GPU) and Cpu has no device at all.
    std::shared_ptr<GpuBackend> make_gpu_backend(const io::ic::BackendKind  kind,
                                                 const io::ic::GpuPrecision precision,
                                                 int                        device) {
      switch (kind) {
        case io::ic::BackendKind::Cpu:
          return nullptr;
        case io::ic::BackendKind::Metal:
          if (!MetalBackend::available()) {
            std::cout << "ICLikelihood: Metal backend requested but no device available; using CPU\n";
            return nullptr;
          }
          return std::make_shared<MetalBackend>();
        case io::ic::BackendKind::Cuda: {
          if (!CudaBackend::available()) {
            std::cout << "ICLikelihood: CUDA backend requested but no device available; using CPU\n";
            return nullptr;
          }
          const bool fp64 = precision == io::ic::GpuPrecision::Fp64;
          std::cout << "ICLikelihood: CUDA backend using " << (fp64 ? "FP64" : "FP32") << " kernels on device "
                    << device << '\n';
          return std::make_shared<CudaBackend>(fp64, device);
        }
      }
      return nullptr;
    }

  }  // namespace

  std::shared_ptr<Likelihood> ICExperimentModule::create_likelihood(std::shared_ptr<io::Options> options, int worker_index) {
    // Heavy per-process setup happens once per device, only for the selected
    // experiment: the parquet load (ICDataBase, shared by every device) and
    // each device's context, kernel compiles and MC column uploads
    // (GpuBackend, one per ordinal --gpuDevices assigns this worker to). Both
    // are cached on the module so repeated Fit constructions -- a scan builds
    // one per grid point -- reuse them instead of paying for them again. The
    // lock makes the first construction of each safe when several scan
    // workers build their Fits at the same time; the sample is const
    // afterwards and each backend is internally locked.
    const int device = m_InputOptions->backend_kind() == io::ic::BackendKind::Cuda
                          ? options->inputOptions().gpu_device_for_worker(worker_index)
                          : 0;

    std::shared_ptr<const io::ic::ICDataBase> database;
    std::shared_ptr<GpuBackend>               gpu;
    {
      const std::scoped_lock lock(m_CacheMutex);
      if (m_DataBase == nullptr) {
        m_DataBase = std::make_shared<const io::ic::ICDataBase>(m_InputOptions->samples());
      }
      auto& cached = m_GpuBackends[device];
      if (cached == nullptr) {
        cached = make_gpu_backend(m_InputOptions->backend_kind(), m_InputOptions->gpu_precision(), device);
      }
      database = m_DataBase;
      gpu      = cached;
    }

    return std::make_shared<ICLikelihood>(std::move(options),
                                          std::move(database),
                                          *m_InputOptions,
                                          std::move(gpu));
  }

  void ICExperimentModule::write_results(Fit& fit, std::string_view name) {
    // The likelihood comes from the Fit being written, not from the module:
    // concurrent scan workers each own one, and a module-wide handle would let
    // them write each other's results.
    const auto likelihood = std::dynamic_pointer_cast<ICLikelihood>(fit.likelihood());
    if (likelihood == nullptr) {
      throw std::logic_error("ICExperimentModule::write_results called with a fit that holds no ICLikelihood");
    }
    result::ic::write_ice_cube_results(fit, *likelihood, *m_InputOptions, name);
  }

}  // namespace ana::ic
