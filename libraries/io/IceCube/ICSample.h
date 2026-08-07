#pragma once

#include "ICParameter.h"  // params::ic::nBarrParams

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace io::ic {

  /**
   * Struct-of-Arrays storage for one IceCube MC analysis sample.
   * All columns are contiguous vectors for cache-efficient inner loops.
   * Loaded once by ICDataBase at startup; never modified during fitting.
   *
   * Only the columns the sample's declared components need are populated:
   * e_true always, astro_baseline for "astro", and conv/prompt/barr_conv for
   * the atmospheric components. Columns of components the sample does not
   * declare stay empty, and the flux component that would read them is not
   * constructed (see SampleLikelihood).
   *
   * Column meaning (parquet branch in parentheses):
   *   e_true          true neutrino energy in GeV        (MCPrimaryEnergy)
   *   astro_baseline  astrophysical baseline weight       (powerlaw)
   *   conv_baseline   conventional atmo weight, H4a       (mceq_conv_H4a_SIBYLL23c)
   *   conv_alt        conventional atmo weight, GST4      (mceq_conv_GST4_SIBYLL23c)  [CRGrad alt]
   *   prompt_baseline prompt atmo weight, H4a             (mceq_pr_H4a_SIBYLL23c)
   *   prompt_alt      prompt atmo weight, GST4            (mceq_pr_GST4_SIBYLL23c)    [CRGrad alt]
   *   barr_conv[k]    d(conv_flux)/d(barr_k), k in {H,W,Y,Z}  (barr_{h,w,y,z}_mceq_H4a_SIBYLL23c)
   *
   * Reconstructed energy/zenith are only needed to assign each event to an
   * analysis bin; that is done at load time and stored as the CSR bin layout,
   * so no reco columns are retained for the fit loop.
   */
  struct ICSample {
    // --- Per-event columns (fit-time) ---
    std::vector<double> e_true;
    std::vector<double> astro_baseline;
    std::vector<double> conv_baseline;
    std::vector<double> conv_alt;
    std::vector<double> prompt_baseline;
    std::vector<double> prompt_alt;

    // Barr gradients: barr_conv[k][i] = d(conv_flux)/d(barr_k) for event i.
    // NNMFit BarrLinear reweight: conv *= (1 + barr_k * barr_conv[k][i] / conv_baseline[i]).
    std::array<std::vector<double>, params::ic::nBarrParams> barr_conv;

    // Veto passing-fraction coefficients, index order {a, b, c}: NNMFit expands
    // log10(PF) to second order around 100 GeV per event and per component,
    //   log10(PF_i) = a_i + b_i * e + c_i * e^2,
    // with e set by the shared VetoThreshold parameter. Populated only for samples
    // declaring the veto components.
    std::array<std::vector<double>, 3> veto_conv;
    std::array<std::vector<double>, 3> veto_prompt;

    // --- Bin assignment, filled at load time ---
    // bin_idx[i] = flat index in the sample's own Binning, -1 if out of range.
    std::vector<int> bin_idx;

    // Fraction of this bin's nominal MC weight that survived the sample's
    // topology cut, one entry per MC bin; empty when the sample has no cut.
    // Only consumer: DetectorSystematics, which rescales gradients exported from
    // the unfiltered sample. Not a reweighting of the prediction -- the per-event
    // columns above are already cut, so they need no correction.
    std::vector<double> topology_bin_fraction;

    // --- CSR bucket layout, built by sort_into_bins() ---
    // After sorting, events of analysis bin b occupy the contiguous range
    // [bin_offsets[b], bin_offsets[b+1]) in every per-event column, so flux
    // loops run bin-major with a scalar accumulator and parallelise over bins.
    std::vector<std::size_t> bin_offsets;  // size total_bins + 1

    // --- GPU work decomposition, built by sort_into_bins() ---
    // One block per bin is the obvious way to run the flux kernels over the CSR
    // layout above, and it is a bad one: bin populations span orders of
    // magnitude (reco energy falls steeply), so the kernel runs as long as the
    // fattest bin while most blocks retire immediately -- and a sample binned
    // coarsely enough to have a single bin runs on a single SM.
    //
    // So the kernels are dispatched over *chunks* instead. chunk_offsets is a
    // partition of [0, size()) into ranges that never span a bin, each at most
    // kEventsPerChunk long, with at least one (possibly empty) chunk per bin so
    // every bin is always written. bin_chunk_offsets is the CSR index from a bin
    // to its range of chunks. A kernel reduces one chunk to one partial and a
    // second pass sums each bin's partials -- deterministically, unlike
    // atomics, which matters because scan points have to be comparable.
    std::vector<std::size_t> chunk_offsets;      // size n_chunks + 1
    std::vector<std::size_t> bin_chunk_offsets;  // size total_bins + 1

    [[nodiscard]] std::size_t n_chunks() const noexcept {
      return chunk_offsets.empty() ? 0 : chunk_offsets.size() - 1;
    }

    [[nodiscard]] std::size_t size()  const noexcept { return e_true.size(); }
    [[nodiscard]] bool        empty() const noexcept { return e_true.empty(); }

    /**
     * Drop out-of-range events (bin_idx < 0), reorder every per-event column so
     * events are grouped by analysis bin, and build the CSR bin_offsets index.
     * Call once at load time after all columns are populated; never during the fit.
     * `total_bins` is the flattened bin count of the sample's own Binning.
     */
    void sort_into_bins(int total_bins) {
      const std::size_t N = size();

      // Permutation of the valid events (bin_idx >= 0), grouped by bin.
      // stable_sort keeps the original within-bin order for reproducibility.
      std::vector<std::size_t> perm;
      perm.reserve(N);
      for (std::size_t i = 0; i < N; ++i)
        if (bin_idx[i] >= 0) perm.push_back(i);
      std::ranges::stable_sort(perm, [this](const std::size_t a, const std::size_t b) {
        return bin_idx[a] < bin_idx[b];
      });

      // Apply the permutation to one column (materialises a compacted copy).
      // A column left empty by ICDataBase -- because the sample does not declare
      // the component that needs it -- is skipped rather than indexed.
      auto reorder = [&perm](auto& col) {
        if (col.empty()) return;
        using Column = std::decay_t<decltype(col)>;
        Column out;
        out.reserve(perm.size());
        for (std::size_t p : perm) out.push_back(std::move(col[p]));
        col = std::move(out);
      };

      reorder(e_true);
      reorder(astro_baseline);
      reorder(conv_baseline);
      reorder(conv_alt);
      reorder(prompt_baseline);
      reorder(prompt_alt);
      for (auto& grad : barr_conv) reorder(grad);
      for (auto& coefficient : veto_conv) reorder(coefficient);
      for (auto& coefficient : veto_prompt) reorder(coefficient);
      reorder(bin_idx);

      // CSR prefix sum over the now-sorted, valid events.
      bin_offsets.assign(total_bins + 1, 0);
      for (int b : bin_idx) ++bin_offsets[b + 1];
      for (int b = 0; b < total_bins; ++b) bin_offsets[b + 1] += bin_offsets[b];

      build_chunks(total_bins);
    }

    /**
     * Default chunk size. A compromise: small enough that thousands of blocks
     * are in flight even for a sample whose binning has very few bins, large
     * enough that each block's tree reduction is amortised over real work.
     * 8192 events per chunk is 32 events per thread at 256 threads.
     */
    static constexpr std::size_t kDefaultEventsPerChunk = 8192;

    /**
     * Split each bin's CSR range into chunks of at most `events_per_chunk`
     * events, filling chunk_offsets / bin_chunk_offsets. Called by
     * sort_into_bins(); separate so the invariants are stated in one place and
     * so tests can force many chunks per bin without a huge sample.
     *
     * Postconditions: chunk_offsets is ascending and partitions
     * [0, bin_offsets.back()); no chunk spans a bin boundary; every bin owns at
     * least one (possibly empty) chunk, so every bin is written by the gather.
     */
    void build_chunks(int total_bins, std::size_t events_per_chunk = kDefaultEventsPerChunk) {
      const std::size_t kEventsPerChunk = std::max<std::size_t>(1, events_per_chunk);

      bin_chunk_offsets.assign(total_bins + 1, 0);
      chunk_offsets.clear();
      chunk_offsets.push_back(0);

      for (int b = 0; b < total_bins; ++b) {
        const std::size_t start = bin_offsets[b];
        const std::size_t end   = bin_offsets[b + 1];
        const std::size_t n     = end - start;
        // max(1, ...): an empty bin still gets one empty chunk, so bin_gather
        // writes a 0 into it rather than leaving the previous value in place.
        const std::size_t n_chunk = std::max<std::size_t>(1, (n + kEventsPerChunk - 1) / kEventsPerChunk);
        for (std::size_t c = 1; c <= n_chunk; ++c)
          chunk_offsets.push_back(start + std::min(n, c * kEventsPerChunk));
        bin_chunk_offsets[b + 1] = bin_chunk_offsets[b] + n_chunk;
      }
    }
  };

}  // namespace io::ic
