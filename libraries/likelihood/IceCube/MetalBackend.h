#pragma once

#include "GpuBackend.h"

#include <cstddef>
#include <memory>

namespace ana::ic {

  class MetalSession;

  /**
   * Shared Apple-Metal compute backend for the IceCube per-event flux loops.
   *
   * Owns the single MTLDevice, the kernel-pipeline cache and the uploaded MC
   * column buffers; per-sample state lives on the MetalSession objects it hands
   * out. The flux components (PowerlawFlux, AtmosphericFlux, and any future
   * per-event reweight) drive a session through the pure-C++ GpuSession facade,
   * so the Metal and Objective-C types stay confined to MetalBackend.mm. A
   * non-Apple stub (MetalBackend_stub.cpp) provides the same symbols with
   * available()==false so the library builds and links everywhere.
   *
   * Usage (one backend per process, one session per sample):
   *   auto b = std::make_shared<MetalBackend>();       // must be shared_ptr
   *   auto s = b->create_session();
   *   s->ensure_kernel("powerlaw_hist", kSource);      // idempotent, shared
   *   int et  = s->upload_column(sample.e_true.data(), M);  // deduped by pointer
   *   int off = s->upload_offsets(sample.bin_offsets.data(), nBins+1);
   *   int h   = s->alloc_output(nBins);                // private to this session
   *   int pe  = s->alloc_output(M);                    // or -1 if unused
   *   ...
   *   s->dispatch("powerlaw_hist", {et,base,off}, &params, sizeof(params), h, pe);
   *   const float* hist = s->contents(h);
   *
   * Precision: FP32 weights + threadgroup tree reduction (validated ~5e-7/bin).
   */
  class MetalBackend final : public GpuBackend {
   public:
    MetalBackend();  // throws std::runtime_error if no Metal device is present
    ~MetalBackend() override;

    MetalBackend(const MetalBackend&)            = delete;
    MetalBackend& operator=(const MetalBackend&) = delete;

    /** True if a usable Metal device exists. Cheap; call before constructing. */
    [[nodiscard]] static bool available() noexcept;

    [[nodiscard]] GpuLanguage language() const noexcept override { return GpuLanguage::Metal; }

    [[nodiscard]] std::shared_ptr<GpuSession> create_session() override;

   private:
    friend class MetalSession;
    void* m_State = nullptr;  // opaque MetalState* (Obj-C++), nullptr in the stub
  };

  /**
   * One sample's view of a MetalBackend: owns that sample's output buffers and
   * the handle table, forwards kernel compiles and column uploads to the shared
   * backend. See GpuSession for the handle-space contract.
   */
  class MetalSession final : public GpuSession {
   public:
    explicit MetalSession(std::shared_ptr<MetalBackend> backend);
    ~MetalSession() override;

    MetalSession(const MetalSession&)            = delete;
    MetalSession& operator=(const MetalSession&) = delete;

    [[nodiscard]] GpuLanguage language() const noexcept override { return GpuLanguage::Metal; }

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
    std::shared_ptr<MetalBackend> m_Backend;  // keeps the shared cache alive
    void*                         m_State = nullptr;  // opaque MetalSessionState*
  };

}  // namespace ana::ic
