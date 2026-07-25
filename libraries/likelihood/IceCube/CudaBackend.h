#pragma once

#include "GpuBackend.h"

#include <cstddef>

namespace ana::ic {

  /**
   * NVIDIA-CUDA implementation of GpuBackend, mirroring MetalBackend. Kernels are
   * runtime-compiled with NVRTC (the analogue of Metal's newLibraryWithSource)
   * and launched through the CUDA driver API, so the flux components drive it
   * through the same pure-C++ facade and the CUDA headers stay confined to
   * CudaBackend.cpp.
   *
   * Unlike Apple unified memory, discrete NVIDIA device memory is not host
   * mapped, so each output buffer keeps a host mirror that dispatch() refreshes
   * (device -> host) after synchronising; contents() then returns that CPU
   * pointer exactly like the Metal shared-memory path. Input columns live only
   * on the device.
   *
   * A non-CUDA stub (CudaBackend_stub.cpp) provides the same symbols with
   * available()==false so the library builds and links where no CUDA toolkit is
   * present.
   */
  class CudaBackend final : public GpuBackend {
   public:
    CudaBackend();  // throws std::runtime_error if no CUDA device is present
    ~CudaBackend() override;

    CudaBackend(const CudaBackend&)            = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    /** True if a usable CUDA device exists. Cheap; call before constructing. */
    [[nodiscard]] static bool available() noexcept;

    [[nodiscard]] GpuLanguage language() const noexcept override { return GpuLanguage::Cuda; }

    void ensure_kernel(const char* name, const char* source) override;
    int  upload_column(const double* data, std::size_t n) override;
    int  upload_offsets(const std::size_t* data, std::size_t n) override;
    int  alloc_output(std::size_t n) override;
    void dispatch(const char* name,
                  const int*  inputs,
                  int         n_inputs,
                  const void* params,
                  std::size_t params_len,
                  int         hist,
                  int         per_event,
                  std::size_t n_groups) override;
    [[nodiscard]] const float* contents(int handle) const noexcept override;

   private:
    void* m_State = nullptr;  // opaque CudaState*, nullptr in the stub
  };

}  // namespace ana::ic
