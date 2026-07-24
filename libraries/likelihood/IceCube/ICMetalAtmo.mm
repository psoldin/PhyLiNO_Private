// Metal implementation of the conventional + prompt atmospheric flux histogram.
// Compiled only on Apple platforms (guarded in CMakeLists.txt), with ARC.
//
// One threadgroup per analysis bin over the CSR-sorted ICSample, mirroring
// AtmosphericFlux::recalculate: CRGrad blend, conventional Barr product over the
// 4 params, conv+prompt with a shared DeltaGamma tilt, grid-stride sum + tree
// reduction. The same-event conv+prompt sum is what feeds the SAY ssq term.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include "ICMetalAtmo.h"

#include "../../io/IceCube/ICConstants.h"
#include "../../io/IceCube/ICParameter.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ana::ic {

  namespace {

    constexpr std::uint32_t kThreadsPerGroup = 256;  // must match the kernel below
    static_assert(params::ic::nBarrParams == 4, "atmo kernel unrolls exactly 4 Barr params");

    // Matches AtmoParams in the kernel.
    struct AtmoParams {
      float cr;
      float dg;
      float conv_norm;
      float prompt_norm;
      float barr0;
      float barr1;
      float barr2;
      float barr3;
      float inv_eref_conv;
      float inv_eref_prompt;
      int   write_pe;
    };

    NSString* kernel_source() {
      return @R"METAL(
        #include <metal_stdlib>
        using namespace metal;

        struct AtmoParams {
          float cr; float dg; float conv_norm; float prompt_norm;
          float barr0; float barr1; float barr2; float barr3;
          float inv_eref_conv; float inv_eref_prompt;
          int write_pe;
        };
        constant uint kThreadsPerGroup = 256;

        kernel void atmo_hist(
            device const float*   e_true      [[buffer(0)]],
            device const float*   conv_base   [[buffer(1)]],
            device const float*   conv_alt    [[buffer(2)]],
            device const float*   prompt_base [[buffer(3)]],
            device const float*   prompt_alt  [[buffer(4)]],
            device const float*   barr0       [[buffer(5)]],
            device const float*   barr1       [[buffer(6)]],
            device const float*   barr2       [[buffer(7)]],
            device const float*   barr3       [[buffer(8)]],
            device const uint*    bin_offsets [[buffer(9)]],
            constant AtmoParams&  p           [[buffer(10)]],
            device float*         hist        [[buffer(11)]],
            device float*         per_event   [[buffer(12)]],
            uint bin [[threadgroup_position_in_grid]],
            uint tid [[thread_position_in_threadgroup]])
        {
          const uint start = bin_offsets[bin];
          const uint end   = bin_offsets[bin + 1];
          float acc = 0.0f;
          for (uint i = start + tid; i < end; i += kThreadsPerGroup) {
            const float et = e_true[i];
            float event_total = 0.0f;

            const float cb = conv_base[i];
            if (cb > 0.0f) {
              float cw = cb + p.cr * (conv_alt[i] - cb);
              cw *= 1.0f + p.barr0 * barr0[i] / cb;
              cw *= 1.0f + p.barr1 * barr1[i] / cb;
              cw *= 1.0f + p.barr2 * barr2[i] / cb;
              cw *= 1.0f + p.barr3 * barr3[i] / cb;
              cw *= p.conv_norm * pow(et * p.inv_eref_conv, -p.dg);
              event_total += cw;
            }

            const float pb = prompt_base[i];
            if (pb > 0.0f) {
              float pw = pb + p.cr * (prompt_alt[i] - pb);
              pw *= p.prompt_norm * pow(et * p.inv_eref_prompt, -p.dg);
              event_total += pw;
            }

            if (p.write_pe) per_event[i] = event_total;
            acc += event_total;
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
      id<MTLBuffer>               conv_base;
      id<MTLBuffer>               conv_alt;
      id<MTLBuffer>               prompt_base;
      id<MTLBuffer>               prompt_alt;
      id<MTLBuffer>               barr[params::ic::nBarrParams];
      id<MTLBuffer>               off;
      id<MTLBuffer>               hist;
      id<MTLBuffer>               pe;
      std::uint32_t               M;
      double                      convERef;
      double                      promptERef;
    };

    id<MTLBuffer> upload_f32(id<MTLDevice> dev, const std::vector<double>& col, std::uint32_t M) {
      std::vector<float> f(M);
      for (std::uint32_t i = 0; i < M; ++i) f[i] = static_cast<float>(col[i]);
      return [dev newBufferWithBytes:f.data() length:M * sizeof(float)
                             options:MTLResourceStorageModeShared];
    }

  }  // namespace

  ICMetalAtmo::ICMetalAtmo(const io::ic::ICSample& sample,
                           const double            conv_delta_gamma_e_ref,
                           const double            prompt_delta_gamma_e_ref) {
    @autoreleasepool {
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      if (!dev)
        throw std::runtime_error("ICMetalAtmo: no Metal device available");

      NSError*       err = nil;
      id<MTLLibrary> lib = [dev newLibraryWithSource:kernel_source()
                                             options:[MTLCompileOptions new]
                                               error:&err];
      if (!lib)
        throw std::runtime_error(std::string("ICMetalAtmo: kernel compile failed: ") +
                                 (err ? err.localizedDescription.UTF8String : "unknown"));

      id<MTLFunction>             fn  = [lib newFunctionWithName:@"atmo_hist"];
      id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
      if (!pso)
        throw std::runtime_error(std::string("ICMetalAtmo: pipeline creation failed: ") +
                                 (err ? err.localizedDescription.UTF8String : "unknown"));

      const std::uint32_t M = static_cast<std::uint32_t>(sample.size());

      auto* s        = new MetalState;
      s->dev         = dev;
      s->queue       = [dev newCommandQueue];
      s->pso         = pso;
      s->M           = M;
      s->convERef    = conv_delta_gamma_e_ref;
      s->promptERef  = prompt_delta_gamma_e_ref;

      s->et          = upload_f32(dev, sample.e_true,          M);
      s->conv_base   = upload_f32(dev, sample.conv_baseline,   M);
      s->conv_alt    = upload_f32(dev, sample.conv_alt,        M);
      s->prompt_base = upload_f32(dev, sample.prompt_baseline, M);
      s->prompt_alt  = upload_f32(dev, sample.prompt_alt,      M);
      for (int k = 0; k < params::ic::nBarrParams; ++k)
        s->barr[k] = upload_f32(dev, sample.barr_conv[k], M);

      std::vector<std::uint32_t> off_u(sample.bin_offsets.begin(), sample.bin_offsets.end());
      const auto shared = MTLResourceStorageModeShared;
      s->off  = [dev newBufferWithBytes:off_u.data() length:off_u.size() * sizeof(std::uint32_t) options:shared];
      s->hist = [dev newBufferWithLength:io::ic::Constants::nBins * sizeof(float) options:shared];
      s->pe   = [dev newBufferWithLength:M * sizeof(float) options:shared];

      m_State = s;
    }
  }

  ICMetalAtmo::~ICMetalAtmo() {
    delete static_cast<MetalState*>(m_State);
  }

  void ICMetalAtmo::recalculate(const double  cr,
                                const double  delta_gamma,
                                const double  conv_norm,
                                const double  prompt_norm,
                                const double* barr,
                                const bool    fill_per_event) {
    auto* s = static_cast<MetalState*>(m_State);
    @autoreleasepool {
      AtmoParams p;
      p.cr              = static_cast<float>(cr);
      p.dg              = static_cast<float>(delta_gamma);
      p.conv_norm       = static_cast<float>(conv_norm);
      p.prompt_norm     = static_cast<float>(prompt_norm);
      p.barr0           = static_cast<float>(barr[0]);
      p.barr1           = static_cast<float>(barr[1]);
      p.barr2           = static_cast<float>(barr[2]);
      p.barr3           = static_cast<float>(barr[3]);
      p.inv_eref_conv   = static_cast<float>(1.0 / s->convERef);
      p.inv_eref_prompt = static_cast<float>(1.0 / s->promptERef);
      p.write_pe        = fill_per_event ? 1 : 0;

      id<MTLCommandBuffer>         cb  = [s->queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      [enc setComputePipelineState:s->pso];
      [enc setBuffer:s->et          offset:0 atIndex:0];
      [enc setBuffer:s->conv_base   offset:0 atIndex:1];
      [enc setBuffer:s->conv_alt    offset:0 atIndex:2];
      [enc setBuffer:s->prompt_base offset:0 atIndex:3];
      [enc setBuffer:s->prompt_alt  offset:0 atIndex:4];
      [enc setBuffer:s->barr[0]     offset:0 atIndex:5];
      [enc setBuffer:s->barr[1]     offset:0 atIndex:6];
      [enc setBuffer:s->barr[2]     offset:0 atIndex:7];
      [enc setBuffer:s->barr[3]     offset:0 atIndex:8];
      [enc setBuffer:s->off         offset:0 atIndex:9];
      [enc setBytes:&p length:sizeof(p) atIndex:10];
      [enc setBuffer:s->hist        offset:0 atIndex:11];
      [enc setBuffer:s->pe          offset:0 atIndex:12];
      [enc dispatchThreadgroups:MTLSizeMake(io::ic::Constants::nBins, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(kThreadsPerGroup, 1, 1)];
      [enc endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
    }
  }

  const float* ICMetalAtmo::histogram() const noexcept {
    return static_cast<const float*>(static_cast<MetalState*>(m_State)->hist.contents);
  }

  const float* ICMetalAtmo::per_event_weight() const noexcept {
    return static_cast<const float*>(static_cast<MetalState*>(m_State)->pe.contents);
  }

  std::size_t ICMetalAtmo::size() const noexcept {
    return static_cast<MetalState*>(m_State)->M;
  }

  bool ICMetalAtmo::available() noexcept {
    @autoreleasepool {
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      return dev != nil;
    }
  }

}  // namespace ana::ic
