// Metal implementation of the astrophysical power-law flux histogram.
// Compiled only on Apple platforms (guarded in CMakeLists.txt), with ARC.
//
// Kernel: one threadgroup per analysis bin. Events are already CSR-sorted by bin
// in ICSample, so bin b owns [off[b], off[b+1]) in every column. Each threadgroup
// sums its bin with a grid-stride loop and a threadgroup tree reduction.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "ICMetalPowerlaw.h"

#include "../../io/IceCube/ICConstants.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ana::ic {

  namespace {

    constexpr std::uint32_t kThreadsPerGroup = 256;  // must match the kernel below

    // Matches PowerlawParams in the kernel (std140-friendly: 3 floats + int).
    struct PowerlawParams {
      float eff_norm;
      float inv_eref;
      float exponent;
      int   write_pe;
    };

    NSString* kernel_source() {
      return @R"METAL(
        #include <metal_stdlib>
        using namespace metal;

        struct PowerlawParams { float eff_norm; float inv_eref; float exponent; int write_pe; };
        constant uint kThreadsPerGroup = 256;

        kernel void powerlaw_hist(
            device const float*      e_true      [[buffer(0)]],
            device const float*      baseline    [[buffer(1)]],
            device const uint*       bin_offsets [[buffer(2)]],
            constant PowerlawParams& p           [[buffer(3)]],
            device float*            hist        [[buffer(4)]],
            device float*            per_event   [[buffer(5)]],
            uint bin [[threadgroup_position_in_grid]],
            uint tid [[thread_position_in_threadgroup]])
        {
          const uint start = bin_offsets[bin];
          const uint end   = bin_offsets[bin + 1];
          float acc = 0.0f;
          for (uint i = start + tid; i < end; i += kThreadsPerGroup) {
            const float w = baseline[i] * p.eff_norm * pow(e_true[i] * p.inv_eref, p.exponent);
            if (p.write_pe) per_event[i] = w;
            acc += w;
          }
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
    }

    struct MetalState {
      id<MTLDevice>               dev;
      id<MTLCommandQueue>         queue;
      id<MTLComputePipelineState> pso;
      id<MTLBuffer>               et;
      id<MTLBuffer>               base;
      id<MTLBuffer>               off;
      id<MTLBuffer>               hist;
      id<MTLBuffer>               pe;
      std::uint32_t               M;
      double                      eRef;
      double                      refIndex;
      bool                        perType;
    };

  }  // namespace

  ICMetalPowerlaw::ICMetalPowerlaw(const io::ic::ICSample& sample,
                                   const double            e_ref_gev,
                                   const double            reference_index,
                                   const bool              per_type_norm) {
    @autoreleasepool {
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      if (!dev)
        throw std::runtime_error("ICMetalPowerlaw: no Metal device available");

      NSError*       err  = nil;
      id<MTLLibrary> lib  = [dev newLibraryWithSource:kernel_source()
                                              options:[MTLCompileOptions new]
                                                error:&err];
      if (!lib)
        throw std::runtime_error(std::string("ICMetalPowerlaw: kernel compile failed: ") +
                                 (err ? err.localizedDescription.UTF8String : "unknown"));

      id<MTLFunction>             fn  = [lib newFunctionWithName:@"powerlaw_hist"];
      id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
      if (!pso)
        throw std::runtime_error(std::string("ICMetalPowerlaw: pipeline creation failed: ") +
                                 (err ? err.localizedDescription.UTF8String : "unknown"));

      const std::uint32_t M = static_cast<std::uint32_t>(sample.size());

      // FP32 copies of the per-event columns (already CSR bin-major in ICSample).
      std::vector<float> et_f(M), base_f(M);
      for (std::uint32_t i = 0; i < M; ++i) {
        et_f[i]   = static_cast<float>(sample.e_true[i]);
        base_f[i] = static_cast<float>(sample.astro_baseline[i]);
      }
      std::vector<std::uint32_t> off_u(sample.bin_offsets.begin(), sample.bin_offsets.end());

      auto* s      = new MetalState;
      s->dev       = dev;
      s->queue     = [dev newCommandQueue];
      s->pso       = pso;
      s->M         = M;
      s->eRef      = e_ref_gev;
      s->refIndex  = reference_index;
      s->perType   = per_type_norm;

      const auto shared = MTLResourceStorageModeShared;
      s->et   = [dev newBufferWithBytes:et_f.data()   length:M * sizeof(float)              options:shared];
      s->base = [dev newBufferWithBytes:base_f.data() length:M * sizeof(float)              options:shared];
      s->off  = [dev newBufferWithBytes:off_u.data()  length:off_u.size() * sizeof(std::uint32_t) options:shared];
      s->hist = [dev newBufferWithLength:io::ic::Constants::nBins * sizeof(float)           options:shared];
      s->pe   = [dev newBufferWithLength:M * sizeof(float)                                   options:shared];

      m_State = s;
    }
  }

  ICMetalPowerlaw::~ICMetalPowerlaw() {
    delete static_cast<MetalState*>(m_State);
  }

  void ICMetalPowerlaw::recalculate(const double norm, const double gamma, const bool fill_per_event) {
    auto* s = static_cast<MetalState*>(m_State);
    @autoreleasepool {
      PowerlawParams p;
      p.eff_norm = static_cast<float>(s->perType ? norm : 0.5 * norm);
      p.inv_eref = static_cast<float>(1.0 / s->eRef);
      p.exponent = static_cast<float>(s->refIndex - gamma);
      p.write_pe = fill_per_event ? 1 : 0;

      id<MTLCommandBuffer>         cb  = [s->queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:s->pso];
      [enc setBuffer:s->et   offset:0 atIndex:0];
      [enc setBuffer:s->base offset:0 atIndex:1];
      [enc setBuffer:s->off  offset:0 atIndex:2];
      [enc setBytes:&p length:sizeof(p) atIndex:3];
      [enc setBuffer:s->hist offset:0 atIndex:4];
      [enc setBuffer:s->pe   offset:0 atIndex:5];
      [enc dispatchThreadgroups:MTLSizeMake(io::ic::Constants::nBins, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }
  }

  const float* ICMetalPowerlaw::histogram() const noexcept {
    return static_cast<const float*>(static_cast<MetalState*>(m_State)->hist.contents);
  }

  const float* ICMetalPowerlaw::per_event_weight() const noexcept {
    return static_cast<const float*>(static_cast<MetalState*>(m_State)->pe.contents);
  }

  std::size_t ICMetalPowerlaw::size() const noexcept {
    return static_cast<MetalState*>(m_State)->M;
  }

  bool ICMetalPowerlaw::available() noexcept {
    @autoreleasepool {
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      return dev != nil;
    }
  }

}  // namespace ana::ic
