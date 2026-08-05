#pragma once

#include <cstddef>
#include <memory>
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
   * Pure-C++ facade over the GPU work of one IceCube sample. MetalSession
   * (Apple) and CudaSession (NVIDIA) implement it; the concrete Metal / Obj-C++
   * and CUDA driver types stay confined to their translation units.
   *
   * A session is what the flux components hold. It owns everything private to
   * one sample -- its output buffers and, from the stream phase on, its own
   * command stream/queue -- while the resources that are identical across every
   * sample and every fit (device, compiled kernels, uploaded MC columns) live on
   * the shared GpuBackend behind it. ensure_kernel/upload_column/upload_offsets
   * forward there and are deduplicated process-wide; alloc_output allocates here
   * and is freed when the session dies.
   *
   * Handles from both sides index one flat table owned by the session: a
   * forwarded upload registers the shared buffer as a non-owning alias row, an
   * alloc_output adds an owning row. Callers therefore never have to know which
   * side a handle came from -- which matters because they mix: SampleLikelihood
   * binds two per-event output buffers and the shared bin-offsets column in a
   * single inputs[] array when it dispatches say_ssq.
   *
   * Only per-event-reduce-to-histogram components belong on a session. Per-bin
   * components (TemplateFlux, DetectorSystematics: O(nBins) work) stay on the
   * CPU; GPU launch overhead would exceed their compute.
   *
   * Buffer-binding convention every kernel must follow: inputs at indices
   * 0..n_inputs-1 (bin_offsets last), the params struct at n_inputs, the
   * histogram at n_inputs+1, and the optional per-event buffer at n_inputs+2.
   * The kernel takes the params struct by value / as a constant buffer, in that
   * same position. Precision: FP32 weights + block/threadgroup tree reduction,
   * unless the backend is_fp64().
   *
   * A session is owned by one SampleLikelihood and is not thread-safe; two
   * samples that run concurrently hold two sessions.
   */
  class GpuSession {
   public:
    virtual ~GpuSession() = default;

    /** Source dialect ensure_kernel() expects. Mirrors the backend's. */
    [[nodiscard]] virtual GpuLanguage language() const noexcept = 0;

    /** True when the backend computes in FP64 (double) instead of FP32. Only
        CUDA can return true -- Apple GPUs have no double support, so Metal is
        always FP32. Flux components read this to pick the FP64 kernel dialect,
        build a double-typed params struct, and read results back via
        contents_f64(). */
    [[nodiscard]] virtual bool is_fp64() const noexcept { return false; }

    /** Compile + cache a compute pipeline for `name` from `source` on the shared
        backend. Idempotent process-wide: the second call with the same name is a
        no-op, whichever session makes it. */
    virtual void ensure_kernel(const char* name, const char* source) = 0;

    /** Upload a per-event double column to the shared backend (converted to FP32
        unless is_fp64()). Identical source pointers are deduplicated to one
        buffer across every session. Returns a handle into this session's table. */
    virtual int upload_column(const double* data, std::size_t n) = 0;

    /** Upload CSR bin offsets as uint32 (deduplicated like columns). */
    virtual int upload_offsets(const std::size_t* data, std::size_t n) = 0;

    /** Allocate a zeroed output buffer of n scalars, owned by this session and
        freed with it. Never deduplicated: two samples writing one histogram
        would corrupt each other. Returns a handle. */
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

    /** FP64 analogue of contents(): CPU-readable doubles of an output buffer,
        valid after the producing dispatch. Only meaningful on an FP64 backend;
        returns nullptr otherwise. */
    [[nodiscard]] virtual const double* contents_f64(int /*handle*/) const noexcept { return nullptr; }
  };

  /**
   * The process-wide half of the GPU facade: one device/context, one compiled
   * kernel cache, one copy of every uploaded MC column. MetalBackend and
   * CudaBackend implement it.
   *
   * A backend is created once (ICExperimentModule caches it beside the
   * ICDataBase whose columns it uploads) and hands out one GpuSession per
   * sample. That split is the point of the type: a Fit is built per scan point,
   * so anything a fit owns is paid for thousands of times, while the columns and
   * the NVRTC/MSL compiles are identical every time and must be paid once.
   *
   * Backends must be held by shared_ptr -- create_session() keeps the backend
   * alive for as long as any session it produced. The backend must in turn not
   * outlive the ICDataBase it uploaded from: its column cache is keyed on the
   * raw column pointers.
   *
   * create_session() and the warmup paths behind it are safe to call from
   * several threads at once (scan workers build their Fits concurrently); the
   * hot path on the returned session takes no shared lock.
   */
  class GpuBackend : public std::enable_shared_from_this<GpuBackend> {
   public:
    virtual ~GpuBackend() = default;

    /** Source dialect the sessions' ensure_kernel() expects. */
    [[nodiscard]] virtual GpuLanguage language() const noexcept = 0;

    /** True when this backend computes in FP64 (double) instead of FP32. */
    [[nodiscard]] virtual bool is_fp64() const noexcept { return false; }

    /** A new session over this backend, for one sample. */
    [[nodiscard]] virtual std::shared_ptr<GpuSession> create_session() = 0;

    /** Distinct MC columns currently uploaded to the device. Grows only when a
        session uploads a column no other session has. */
    [[nodiscard]] virtual std::size_t column_count() const noexcept = 0;

    /** Kernels compiled so far. Grows only on the first ensure_kernel for a
        given name, whichever session makes it. */
    [[nodiscard]] virtual std::size_t kernel_compile_count() const noexcept = 0;

    /** Output buffers currently alive across every session over this backend.
        Must return to its previous value when a session is destroyed -- with the
        backend outliving the fits, an output that is never freed is a leak that
        grows once per scan point.

        These three counters exist for the tests: deduplication and freeing are
        the properties this split is for, and neither is otherwise observable. */
    [[nodiscard]] virtual std::size_t live_output_count() const noexcept = 0;
  };

}  // namespace ana::ic
