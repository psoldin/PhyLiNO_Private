#pragma once

#include <cstddef>
#include <string>

namespace ana::ic {

  /**
   * Build a complete CUDA-C kernel source from a body written against a generic
   * scalar type `real` and a power macro `RPOW`, by prefixing the typedef that
   * selects the precision. One body then serves both the FP32 and FP64 CUDA
   * paths, so the two can never drift apart. (Metal is FP32-only and keeps its
   * own float source.)
   */
  inline std::string cuda_kernel_source(bool fp64, const char* body) {
    return std::string(fp64 ? "typedef double real;\n#define RPOW pow\n"
                            : "typedef float real;\n#define RPOW powf\n") +
           body;
  }

  /**
   * Kernel-source dialect a backend consumes. Flux components ship one source
   * string per dialect (MSL and CUDA C) and hand ensure_kernel() the one that
   * matches the active backend's language().
   */
  enum class GpuLanguage { Metal, Cuda };

  /**
   * Pure-C++ facade over a GPU compute backend for the IceCube per-event flux
   * loops. MetalBackend (Apple) and CudaBackend (NVIDIA) implement it; the
   * concrete Metal / Obj-C++ and CUDA driver types stay confined to their
   * translation units. ICLikelihood picks one backend and shares the single
   * instance across every flux component, so per-event columns (e_true,
   * bin_offsets, ...) are uploaded once.
   *
   * Only per-event-reduce-to-histogram components belong on a backend. Per-bin
   * components (TemplateFlux, DetectorSystematics: O(nBins) work) stay on the
   * CPU; GPU launch overhead would exceed their compute.
   *
   * Buffer-binding convention every kernel must follow: inputs at indices
   * 0..n_inputs-1 (bin_offsets last), the params struct at n_inputs, the
   * histogram at n_inputs+1, and the optional per-event buffer at n_inputs+2.
   * The kernel takes the params struct by value / as a constant buffer, in that
   * same position. Precision: FP32 weights + block/threadgroup tree reduction.
   */
  class GpuBackend {
   public:
    virtual ~GpuBackend() = default;

    /** Source dialect ensure_kernel() expects for this backend. */
    [[nodiscard]] virtual GpuLanguage language() const noexcept = 0;

    /** Compile + cache a compute pipeline for `name` from `source`. Idempotent:
        a second call with the same name is a no-op. */
    virtual void ensure_kernel(const char* name, const char* source) = 0;

    /** Upload an FP32 copy of a per-event double column. Identical source
        pointers are deduplicated to one buffer. Returns a handle. */
    virtual int upload_column(const double* data, std::size_t n) = 0;

    /** Upload CSR bin offsets as uint32 (deduplicated like columns). */
    virtual int upload_offsets(const std::size_t* data, std::size_t n) = 0;

    /** Allocate a zeroed FP32 output buffer of n floats. Returns a handle. */
    virtual int alloc_output(std::size_t n) = 0;

    /** Dispatch n_groups groups of kernel `name`.
        inputs[0..n_inputs) bind at indices 0.., params at n_inputs, hist at
        n_inputs+1, and per_event at n_inputs+2 when >= 0. Blocks. */
    virtual void dispatch(const char* name,
                          const int*  inputs,
                          int         n_inputs,
                          const void* params,
                          std::size_t params_len,
                          int         hist,
                          int         per_event,
                          std::size_t n_groups) = 0;

    /** CPU-readable pointer to an output buffer's contents; valid after the
        dispatch that produced it. FP32 mirror -- use contents_f64() on an FP64
        backend. */
    [[nodiscard]] virtual const float* contents(int handle) const noexcept = 0;

    /** True when this backend computes in FP64 (double) instead of FP32. Only
        the CUDA backend can return true -- Apple GPUs have no double support, so
        Metal is always FP32. Flux components read this to pick the FP64 kernel
        dialect, build a double-typed params struct, and read results back via
        contents_f64(). Defaults to false so Metal and the stubs need no change. */
    [[nodiscard]] virtual bool is_fp64() const noexcept { return false; }

    /** FP64 analogue of contents(): CPU-readable doubles of an output buffer,
        valid after the producing dispatch. Only meaningful on an FP64 backend;
        returns nullptr otherwise. */
    [[nodiscard]] virtual const double* contents_f64(int /*handle*/) const noexcept { return nullptr; }
  };

}  // namespace ana::ic
