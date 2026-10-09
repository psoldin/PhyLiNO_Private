#include "GpuBinReduce.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace ana::ic {

  namespace {

    // Occupies the params slot the GpuSession buffer convention reserves at
    // index n_inputs. n_bins is the grid bound -- redundant with the dispatch
    // size, but it keeps the kernel safe if a caller ever over-dispatches.
    // `offset` selects which of a multi-quantity partial buffer's blocks to sum
    // (see GpuBinReduce's n_quantities), and is 0 for the single-quantity case.
    struct GatherParams {
      int n_bins;
      int offset;
    };

    // One group per bin over the chunk partials. The chunk count per bin is
    // small (ceil(bin population / kEventsPerChunk)), so this is a cheap tail
    // pass; the block-stride loop is only there for the few fat bins whose chunk
    // count exceeds the group size.
    constexpr const char* kGatherMetalBody = R"METAL(
      struct GatherParams { int n_bins; int offset; };

      kernel void bin_gather(
          device const float*  partial           [[buffer(0)]],
          device const uint*   bin_chunk_offsets [[buffer(1)]],
          constant GatherParams& p               [[buffer(2)]],
          device float*        hist              [[buffer(3)]],
          uint bin [[threadgroup_position_in_grid]],
          uint tid [[thread_position_in_threadgroup]])
      {
        if (bin >= (uint)p.n_bins) return;
        const uint start = bin_chunk_offsets[bin];
        const uint end   = bin_chunk_offsets[bin + 1];
        // The stride loop is short for a typical bin but runs into the hundreds
        // for the fat bins that hold most of the sample, so it gets the same
        // compensation as the flux kernels' event loops.
        float acc = 0.0f;
        float cmp = 0.0f;
        for (uint j = start + tid; j < end; j += kThreadsPerGroup)
          neumaier_add(acc, cmp, partial[(uint)p.offset + j]);
        acc += cmp;
        threadgroup float shared[kThreadsPerGroup];
        shared[tid] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = kThreadsPerGroup / 2; s > 0; s >>= 1) {
          if (tid < s) shared[tid] += shared[tid + s];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid == 0) hist[bin] = shared[0];
      }
    )METAL";

    // CUDA-C twin. Written against a generic scalar `real`; cuda_kernel_source()
    // prepends the typedef selecting float or double, so one body serves both
    // precisions. GatherParams is int-only and needs no precision variant.
    constexpr const char* kGatherCudaBody = R"CUDA(
      struct GatherParams { int n_bins; int offset; };

      extern "C" __global__ void bin_gather(
          const real*         partial,
          const unsigned int* bin_chunk_offsets,
          GatherParams        p,
          real*               hist)
      {
        const unsigned int bin = blockIdx.x;
        if (bin >= (unsigned int)p.n_bins) return;
        const unsigned int tid      = threadIdx.x;
        const unsigned int nthreads = blockDim.x;
        const unsigned int start    = bin_chunk_offsets[bin];
        const unsigned int end      = bin_chunk_offsets[bin + 1];
        real acc = 0.0;
        real cmp = 0.0;
        for (unsigned int j = start + tid; j < end; j += nthreads)
          neumaier_add(acc, cmp, partial[(unsigned int)p.offset + j]);
        acc += cmp;
        __shared__ real sdata[256];
        sdata[tid] = acc;
        __syncthreads();
        for (unsigned int s = nthreads / 2; s > 0; s >>= 1) {
          if (tid < s) sdata[tid] += sdata[tid + s];
          __syncthreads();
        }
        if (tid == 0) hist[bin] = sdata[0];
      }
    )CUDA";

    // gather_all(): one thread per (quantity, bin) instead of one group per
    // bin. Thread idx covers quantity idx / n_bins of bin idx % n_bins, so the
    // threads of a group read neighbouring bins' chunk offsets.
    struct GatherAllParams {
      int n_bins;
      int n_chunks;
      int n_quantities;
    };

    constexpr const char* kGatherAllMetalBody = R"METAL(
      struct GatherAllParams { int n_bins; int n_chunks; int n_quantities; };

      kernel void bin_gather_all(
          device const float*     partial           [[buffer(0)]],
          device const uint*      bin_chunk_offsets [[buffer(1)]],
          constant GatherAllParams& p               [[buffer(2)]],
          device float*           table             [[buffer(3)]],
          uint group [[threadgroup_position_in_grid]],
          uint tid [[thread_position_in_threadgroup]])
      {
        const uint idx   = group * kThreadsPerGroup + tid;
        const uint total = (uint)p.n_bins * (uint)p.n_quantities;
        if (idx >= total) return;
        const uint q     = idx / (uint)p.n_bins;
        const uint bin   = idx - q * (uint)p.n_bins;
        const uint start = bin_chunk_offsets[bin];
        const uint end   = bin_chunk_offsets[bin + 1];
        const uint base  = q * (uint)p.n_chunks;
        float acc = 0.0f;
        float cmp = 0.0f;
        for (uint j = start; j < end; ++j)
          neumaier_add(acc, cmp, partial[base + j]);
        table[idx] = acc + cmp;
      }
    )METAL";

    constexpr const char* kGatherAllCudaBody = R"CUDA(
      struct GatherAllParams { int n_bins; int n_chunks; int n_quantities; };

      extern "C" __global__ void bin_gather_all(
          const real*         partial,
          const unsigned int* bin_chunk_offsets,
          GatherAllParams     p,
          real*               table)
      {
        const unsigned int idx   = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int total = (unsigned int)p.n_bins * (unsigned int)p.n_quantities;
        if (idx >= total) return;
        const unsigned int q     = idx / (unsigned int)p.n_bins;
        const unsigned int bin   = idx - q * (unsigned int)p.n_bins;
        const unsigned int start = bin_chunk_offsets[bin];
        const unsigned int end   = bin_chunk_offsets[bin + 1];
        const unsigned int base  = q * (unsigned int)p.n_chunks;
        real acc = 0.0;
        real cmp = 0.0;
        for (unsigned int j = start; j < end; ++j)
          neumaier_add(acc, cmp, partial[base + j]);
        table[idx] = acc + cmp;
      }
    )CUDA";

    // Must match the threadgroup width both backends dispatch with.
    constexpr std::size_t kThreadsPerGroup = 256;

  }  // namespace

  GpuBinReduce::GpuBinReduce(std::shared_ptr<GpuSession> gpu,
                             const io::ic::ICSample&     sample,
                             const std::size_t           n_bins,
                             const std::size_t           n_quantities)
    : m_Gpu(std::move(gpu))
    , m_NChunks(sample.n_chunks())
    , m_NBins(n_bins)
    , m_NQuantities(std::max<std::size_t>(1, n_quantities)) {
    if (!m_Gpu)
      throw std::runtime_error("GpuBinReduce: null session");
    // A mismatch here would silently reduce over the wrong ranges, so it is
    // worth one check at construction: the chunk plan is built by
    // sort_into_bins() for the sample's own MC binning, and every component of a
    // sample must be constructed with that same binning.
    if (sample.bin_chunk_offsets.size() != n_bins + 1)
      throw std::runtime_error("GpuBinReduce: sample chunk plan covers " +
                               std::to_string(sample.bin_chunk_offsets.size()) +
                               " - 1 bins, the component was built for " + std::to_string(n_bins));

    const std::string src =
        gpu_kernel_source(m_Gpu->language(), m_Gpu->is_fp64(), kGatherMetalBody, kGatherCudaBody);
    m_Gpu->ensure_kernel("bin_gather", src.c_str());

    // Both uploads deduplicate against every other component of this sample --
    // and, across fits, against every other session -- because the cache is
    // keyed on ICSample's column pointers and the ICSample outlives the backend.
    m_hChunkOffsets    = m_Gpu->upload_offsets(sample.chunk_offsets.data(), sample.chunk_offsets.size());
    m_hBinChunkOffsets = m_Gpu->upload_offsets(sample.bin_chunk_offsets.data(), sample.bin_chunk_offsets.size());
    // Never read on the host: gather() feeds it straight back into a kernel.
    // One block of n_chunks per reduced quantity, laid end to end.
    m_hPartial = m_Gpu->alloc_output(m_NChunks * m_NQuantities, /*readback=*/false);
  }

  void GpuBinReduce::gather(const int hist, const std::size_t q) const {
    if (q >= m_NQuantities)
      throw std::runtime_error("GpuBinReduce::gather: quantity " + std::to_string(q) +
                               " is out of range, the partial buffer holds " +
                               std::to_string(m_NQuantities));

    const GatherParams p{.n_bins = static_cast<int>(m_NBins),
                         .offset = static_cast<int>(q * m_NChunks)};
    const int          inputs[] = {m_hPartial, m_hBinChunkOffsets};
    m_Gpu->dispatch("bin_gather", inputs, 2, &p, sizeof(p), hist, /*per_event=*/-1, m_NBins);
  }

  void GpuBinReduce::gather_all(const int table) const {
    if (!m_GatherAllReady) {
      const std::string src =
          gpu_kernel_source(m_Gpu->language(), m_Gpu->is_fp64(), kGatherAllMetalBody, kGatherAllCudaBody);
      m_Gpu->ensure_kernel("bin_gather_all", src.c_str());
      m_GatherAllReady = true;
    }

    const GatherAllParams p{.n_bins       = static_cast<int>(m_NBins),
                            .n_chunks     = static_cast<int>(m_NChunks),
                            .n_quantities = static_cast<int>(m_NQuantities)};
    const int             inputs[] = {m_hPartial, m_hBinChunkOffsets};
    const std::size_t     total    = m_NBins * m_NQuantities;
    m_Gpu->dispatch("bin_gather_all", inputs, 2, &p, sizeof(p), table, /*per_event=*/-1,
                    (total + kThreadsPerGroup - 1) / kThreadsPerGroup);
  }

}  // namespace ana::ic
