#pragma once

#include "../../io/IceCube/ICSample.h"
#include "GpuBackend.h"

#include <cstddef>
#include <memory>
#include <span>
#include <type_traits>

namespace ana::ic {

  /**
   * The second half of every per-event GPU reduction in this sample.
   *
   * The flux kernels (powerlaw_hist, atmo_hist) and the SAY ssq kernel all
   * reduce per-event quantities to per-bin ones over the same CSR layout. Doing
   * that with one block per bin makes the kernel as slow as the fattest bin --
   * and a sample whose binning has a single bin runs on a single SM. So they are
   * dispatched over ICSample's chunk decomposition instead (one block per chunk,
   * every chunk the same size bar the tail of each bin), writing one partial per
   * chunk; this class then sums each bin's partials into the histogram.
   *
   * The split costs one extra kernel launch per reduction and buys full
   * occupancy and a balanced grid. It stays deterministic -- the decomposition
   * is fixed at load time and the gather is a fixed tree, so two runs give
   * bit-identical histograms, which atomics into the histogram would not.
   *
   * One instance per reduced quantity: the partial buffer is written by exactly
   * one kernel, so PowerlawFlux, AtmosphericFlux and SampleLikelihood's ssq each
   * own theirs. The uploaded offset arrays are shared -- they are keyed on
   * ICSample's own pointers, which the backend's column cache deduplicates.
   *
   * Usage: bind chunk_offsets() as the kernel's last input (in place of the
   * bin_offsets it used to read), dispatch n_chunks() groups writing to
   * partial() in the histogram slot, then call gather() with the real histogram
   * handle.
   */
  class GpuBinReduce {
   public:
    /**
     * `n_bins` is the sample's MC bin count, and must be the bin count
     * `sample.bin_chunk_offsets` was built for. The session is retained: gather()
     * dispatches on it.
     *
     * `n_quantities` is how many independent per-event quantities the producing
     * kernel reduces in one pass. The partial buffer is that many blocks of
     * n_chunks laid end to end, quantity q occupying [q * n_chunks, (q + 1) *
     * n_chunks); gather(hist, q) sums that block. AtmosphericFlux uses it to
     * bring back the conventional and prompt histograms separately from a
     * single sweep over the events, which is what lets a step in ConvNorm or
     * PromptNorm rescale a cached histogram instead of re-reducing the sample.
     */
    GpuBinReduce(std::shared_ptr<GpuSession> gpu, const io::ic::ICSample& sample, std::size_t n_bins,
                 std::size_t n_quantities = 1);

    /** Group count to dispatch the producing kernel with. */
    [[nodiscard]] std::size_t n_chunks() const noexcept { return m_NChunks; }

    /** Handle of the chunk-boundary array the producing kernel indexes by group. */
    [[nodiscard]] int chunk_offsets() const noexcept { return m_hChunkOffsets; }

    /** Handle of the per-chunk partial buffer the producing kernel writes. */
    [[nodiscard]] int partial() const noexcept { return m_hPartial; }

    /** Sum quantity `q`'s chunk partials per bin into `hist` (an n_bins output handle). */
    void gather(int hist, std::size_t q = 0) const;

   private:
    std::shared_ptr<GpuSession> m_Gpu;
    std::size_t                 m_NChunks          = 0;
    std::size_t                 m_NBins            = 0;
    std::size_t                 m_NQuantities      = 1;
    int                         m_hChunkOffsets    = -1;
    int                         m_hBinChunkOffsets = -1;
    int                         m_hPartial         = -1;
  };

  /**
   * Copy an n_bins output buffer into a host histogram, narrowing from float on
   * an FP32 backend. `R` is the backend's scalar type, i.e. the params struct's
   * Scalar.
   */
  template <class R>
  void read_back_hist(GpuSession& gpu, int hist_handle, std::span<double> histogram) {
    if constexpr (std::is_same_v<R, double>) {
      const double* hist = gpu.contents_f64(hist_handle);
      for (std::size_t bin = 0, n = histogram.size(); bin < n; ++bin) histogram[bin] = hist[bin];
    } else {
      const float* hist = gpu.contents(hist_handle);
      for (std::size_t bin = 0, n = histogram.size(); bin < n; ++bin) histogram[bin] = static_cast<double>(hist[bin]);
    }
  }

  /**
   * Shared tail of every flux kernel's recalculate_gpu(): fill a params struct
   * templated on the FP32/FP64 scalar (ParamsT must expose `using Scalar =
   * R;`), dispatch it over the chunk decomposition, gather the partials into
   * `hist_handle`, then copy the result into `histogram` -- narrowing from
   * float on an FP32 backend. Callers pick ParamsT<double> or ParamsT<float>
   * to match `gpu.is_fp64()`; `fill` fills the params struct passed to it.
   */
  template <class ParamsT, class Fill>
  void dispatch_and_gather_hist(GpuSession& gpu, const char* kernel_name, const int* inputs, int n_inputs,
                                Fill&& fill, const GpuBinReduce& reduce, int per_event_handle,
                                int hist_handle, std::span<double> histogram) {
    ParamsT p{};
    fill(p);
    gpu.dispatch(kernel_name, inputs, n_inputs, &p, sizeof(p), reduce.partial(), per_event_handle,
                 reduce.n_chunks());
    reduce.gather(hist_handle);
    read_back_hist<typename ParamsT::Scalar>(gpu, hist_handle, histogram);
  }

}  // namespace ana::ic
