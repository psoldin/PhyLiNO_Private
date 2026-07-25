#pragma once

#include "GpuBackend.h"

#include <cstddef>

namespace ana::ic {

  /**
   * Shared Apple-Metal compute backend for the IceCube per-event flux loops.
   *
   * Owns the single MTLDevice / command queue / kernel-pipeline cache / GPU
   * buffer cache; the flux components (PowerlawFlux, AtmosphericFlux, and any
   * future per-event reweight) drive it through this pure-C++ facade, so the
   * Metal and Objective-C types stay confined to MetalBackend.mm. A non-Apple
   * stub (MetalBackend_stub.cpp) provides the same symbols with
   * available()==false so the library builds and links everywhere.
   *
   * Only per-event-reduce-to-histogram components belong here. Per-bin
   * components (MuonTemplate, DetectorSystematics: O(nBins) work) stay on the
   * CPU; GPU launch overhead would exceed their compute.
   *
   * Usage (one backend shared by every component):
   *   auto b = std::make_shared<MetalBackend>();
   *   b->ensure_kernel("powerlaw_hist", kSource);           // idempotent
   *   int et  = b->upload_column(sample.e_true.data(), M);  // deduped by pointer
   *   int off = b->upload_offsets(sample.bin_offsets.data(), nBins+1);
   *   int h   = b->alloc_output(nBins);
   *   int pe  = b->alloc_output(M);                          // or -1 if unused
   *   ...
   *   b->dispatch("powerlaw_hist", {et,base,off}, &params, sizeof(params), h, pe);
   *   const float* hist = b->contents(h);
   *
   * Buffer-binding convention the kernels must follow: inputs at indices
   * 0..n_inputs-1 (bin_offsets last), the params struct at n_inputs, the
   * histogram at n_inputs+1, and the optional per-event buffer at n_inputs+2.
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

    /** Compile + cache a compute pipeline for `name` from `source`. Idempotent:
        a second call with the same name is a no-op. */
    void ensure_kernel(const char* name, const char* source) override;

    /** Upload an FP32 copy of a per-event double column. Identical source
        pointers are deduplicated to one shared-memory buffer. Returns a handle. */
    int upload_column(const double* data, std::size_t n) override;

    /** Upload CSR bin offsets as uint32 (deduplicated like columns). */
    int upload_offsets(const std::size_t* data, std::size_t n) override;

    /** Allocate a zeroed FP32 shared-memory output buffer of n floats. */
    int alloc_output(std::size_t n) override;

    /** Dispatch n_groups threadgroups of kernel `name`.
        inputs[0..n_inputs) bind at buffer indices 0.., params at n_inputs,
        hist at n_inputs+1, and per_event at n_inputs+2 when >= 0. Blocks. */
    void dispatch(const char* name,
                  const int*  inputs,
                  int         n_inputs,
                  const void* params,
                  std::size_t params_len,
                  int         hist,
                  int         per_event,
                  std::size_t n_groups) override;

    /** CPU-readable pointer to a buffer's contents (shared/unified memory). */
    [[nodiscard]] const float* contents(int handle) const noexcept override;

   private:
    void* m_State = nullptr;  // opaque MetalState* (Obj-C++), nullptr in the stub
  };

}  // namespace ana::ic
