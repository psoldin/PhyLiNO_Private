#pragma once

#include "GpuBackend.h"

#include <cstddef>
#include <memory>

namespace ana::ic {

  class CudaSession;

  /**
   * NVIDIA-CUDA implementation of GpuBackend, mirroring MetalBackend. Kernels are
   * runtime-compiled with NVRTC (the analogue of Metal's newLibraryWithSource)
   * and launched through the CUDA driver API, so the flux components drive it
   * through the same pure-C++ facade and the CUDA headers stay confined to
   * CudaBackend.cpp.
   *
   * The backend owns the device context, the compiled modules and the uploaded
   * MC columns -- everything identical across samples and fits. One CudaSession
   * per sample owns that sample's output buffers.
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
   *
   * Precision is fixed per instance: fp64==false uses FP32 columns/weights/
   * reduction (the fast default, matching Metal); fp64==true uses FP64
   * throughout (columns, kernels, tree reduction, readback) so the GPU fit
   * reproduces the FP64 CPU reference to full double precision, at the cost of
   * roughly half the arithmetic throughput and double the memory traffic.
   */
  class CudaBackend final : public GpuBackend {
   public:
    // fp64 selects the double-precision compute path (see class comment).
    explicit CudaBackend(bool fp64 = false);  // throws if no CUDA device is present
    ~CudaBackend() override;

    CudaBackend(const CudaBackend&)            = delete;
    CudaBackend& operator=(const CudaBackend&) = delete;

    /** True if a usable CUDA device exists. Cheap; call before constructing. */
    [[nodiscard]] static bool available() noexcept;

    [[nodiscard]] GpuLanguage language() const noexcept override { return GpuLanguage::Cuda; }
    [[nodiscard]] bool        is_fp64() const noexcept override { return m_Fp64; }

    [[nodiscard]] std::shared_ptr<GpuSession> create_session() override;

    [[nodiscard]] std::size_t column_count() const noexcept override;
    [[nodiscard]] std::size_t kernel_compile_count() const noexcept override;
    [[nodiscard]] std::size_t live_output_count() const noexcept override;

   private:
    friend class CudaSession;
    void* m_State = nullptr;   // opaque CudaState*, nullptr in the stub
    bool  m_Fp64  = false;     // FP64 compute path when true
  };

  /**
   * One sample's view of a CudaBackend: owns that sample's output buffers (with
   * their host mirrors) and the handle table, forwards kernel compiles and
   * column uploads to the shared backend. See GpuSession for the handle-space
   * contract.
   */
  class CudaSession final : public GpuSession {
   public:
    explicit CudaSession(std::shared_ptr<CudaBackend> backend);
    ~CudaSession() override;

    CudaSession(const CudaSession&)            = delete;
    CudaSession& operator=(const CudaSession&) = delete;

    [[nodiscard]] GpuLanguage language() const noexcept override { return GpuLanguage::Cuda; }
    [[nodiscard]] bool        is_fp64() const noexcept override;

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
    [[nodiscard]] const float*  contents(int handle) const noexcept override;
    [[nodiscard]] const double* contents_f64(int handle) const noexcept override;

   private:
    std::shared_ptr<CudaBackend> m_Backend;  // keeps the shared cache alive
    void*                        m_State = nullptr;  // opaque CudaSessionState*
  };

}  // namespace ana::ic
