#include "GpuBinReduce.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace ana::ic {

  namespace {

    // Occupies the params slot the GpuSession buffer convention reserves at
    // index n_inputs. bin_gather needs no scalars of its own, so the one field
    // is the grid bound -- redundant with the dispatch size, but it keeps the
    // kernel safe if a caller ever over-dispatches.
    struct GatherParams {
      int n_bins;
    };

    // One group per bin over the chunk partials. The chunk count per bin is
    // small (ceil(bin population / kEventsPerChunk)), so this is a cheap tail
    // pass; the block-stride loop is only there for the few fat bins whose chunk
    // count exceeds the group size.
    constexpr const char* kGatherMetal = R"METAL(
      #include <metal_stdlib>
      using namespace metal;

      struct GatherParams { int n_bins; };
      constant uint kThreadsPerGroup = 256;

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
        float acc = 0.0f;
        for (uint j = start + tid; j < end; j += kThreadsPerGroup) acc += partial[j];
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
      struct GatherParams { int n_bins; };

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
        for (unsigned int j = start + tid; j < end; j += nthreads) acc += partial[j];
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

  }  // namespace

  GpuBinReduce::GpuBinReduce(std::shared_ptr<GpuSession> gpu,
                             const io::ic::ICSample&     sample,
                             const std::size_t           n_bins)
    : m_Gpu(std::move(gpu))
    , m_NChunks(sample.n_chunks())
    , m_NBins(n_bins) {
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

    const std::string cuda_src = m_Gpu->language() == GpuLanguage::Cuda
                                     ? cuda_kernel_source(m_Gpu->is_fp64(), kGatherCudaBody)
                                     : std::string{};
    const char*       src      = m_Gpu->language() == GpuLanguage::Cuda ? cuda_src.c_str() : kGatherMetal;
    m_Gpu->ensure_kernel("bin_gather", src);

    // Both uploads deduplicate against every other component of this sample --
    // and, across fits, against every other session -- because the cache is
    // keyed on ICSample's column pointers and the ICSample outlives the backend.
    m_hChunkOffsets    = m_Gpu->upload_offsets(sample.chunk_offsets.data(), sample.chunk_offsets.size());
    m_hBinChunkOffsets = m_Gpu->upload_offsets(sample.bin_chunk_offsets.data(), sample.bin_chunk_offsets.size());
    // Never read on the host: gather() feeds it straight back into a kernel.
    m_hPartial = m_Gpu->alloc_output(m_NChunks, /*readback=*/false);
  }

  void GpuBinReduce::gather(const int hist) const {
    const GatherParams p{.n_bins = static_cast<int>(m_NBins)};
    const int          inputs[] = {m_hPartial, m_hBinChunkOffsets};
    m_Gpu->dispatch("bin_gather", inputs, 2, &p, sizeof(p), hist, /*per_event=*/-1, m_NBins);
  }

}  // namespace ana::ic
