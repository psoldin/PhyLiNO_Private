#include "PowerlawFlux.h"

#include "../../io/IceCube/ICParameter.h"

#include <cmath>
#include <iostream>

namespace ana::ic {

  namespace {

    // Scalars pushed to the kernel each iteration; matches PowerlawParams below.
    struct PowerlawParams {
      float eff_norm;
      float inv_eref;
      float exponent;
      int   write_pe;
    };

    // One threadgroup per analysis bin over the CSR-sorted sample; grid-stride
    // sum + threadgroup tree reduction. Buffer order matches the MetalBackend
    // convention: inputs (e_true, baseline, bin_offsets), params, hist, per_event.
    constexpr const char* kKernel = R"METAL(
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

  }  // namespace

  PowerlawFlux::PowerlawFlux(const io::ic::ICSample&       sample,
                             const double                  e_ref_gev,
                             const double                  reference_index,
                             const bool                    per_type_norm,
                             std::shared_ptr<MetalBackend> metal,
                             const bool                    need_per_event)
    : m_Sample(sample)
    , m_ERef(e_ref_gev)
    , m_ReferenceIndex(reference_index)
    , m_PerTypeNorm(per_type_norm)
    , m_NeedPerEvent(need_per_event)
    , m_Metal(std::move(metal)) {
    m_Histogram.fill(0.0);
    m_PerEventWeight.assign(sample.size(), 0.0);

    if (m_Metal) {
      const std::size_t M = sample.size();
      m_Metal->ensure_kernel("powerlaw_hist", kKernel);
      m_hETrue    = m_Metal->upload_column(sample.e_true.data(), M);
      m_hBaseline = m_Metal->upload_column(sample.astro_baseline.data(), M);
      m_hOffsets  = m_Metal->upload_offsets(sample.bin_offsets.data(), sample.bin_offsets.size());
      m_hHist     = m_Metal->alloc_output(io::ic::Constants::nBins);
      m_hPerEvent = m_Metal->alloc_output(M);  // always bound; write gated by need
      std::cout << "PowerlawFlux: using Metal GPU backend\n";
    }
  }

  void PowerlawFlux::recalculate(const ParameterWrapper& parameter) noexcept {
    using namespace params::ic;

    const double norm  = parameter[AstroNorm];
    const double gamma = parameter[SpectralIndex];

    if (m_Metal) {
      PowerlawParams p;
      p.eff_norm = static_cast<float>(m_PerTypeNorm ? norm : 0.5 * norm);
      p.inv_eref = static_cast<float>(1.0 / m_ERef);
      p.exponent = static_cast<float>(m_ReferenceIndex - gamma);
      p.write_pe = m_NeedPerEvent ? 1 : 0;

      const int inputs[] = {m_hETrue, m_hBaseline, m_hOffsets};
      m_Metal->dispatch("powerlaw_hist", inputs, 3, &p, sizeof(p), m_hHist, m_hPerEvent);

      const float* hist = m_Metal->contents(m_hHist);
      for (int bin = 0; bin < io::ic::Constants::nBins; ++bin)
        m_Histogram[bin] = static_cast<double>(hist[bin]);
      if (m_NeedPerEvent) {
        const float* pe = m_Metal->contents(m_hPerEvent);
        for (std::size_t i = 0, n = m_PerEventWeight.size(); i < n; ++i)
          m_PerEventWeight[i] = static_cast<double>(pe[i]);
      }
      return;
    }

    // NNMFit Norm with per_type_norm=false halves the per-type normalization.
    const double eff_norm = m_PerTypeNorm ? norm : 0.5 * norm;
    // (E/E_ref)^(ref_index - gamma), baseline is already ~E^(-ref_index).
    const double exponent = m_ReferenceIndex - gamma;

    const auto& off      = m_Sample.bin_offsets;
    const auto& baseline = m_Sample.astro_baseline;
    const auto& e_true   = m_Sample.e_true;

    #pragma omp parallel for
    for (int bin = 0; bin < io::ic::Constants::nBins; ++bin) {
      double acc = 0.0;
      for (std::size_t i = off[bin], n = off[bin + 1]; i < n; ++i) {
        const double w = baseline[i] * eff_norm * std::pow(e_true[i] / m_ERef, exponent);
        acc += w;
        m_PerEventWeight[i] = w;
      }
      m_Histogram[bin] = acc;
    }
  }

  bool PowerlawFlux::check_and_recalculate(const ParameterWrapper& parameter) {
    using namespace params::ic;
    const bool changed = parameter.check_parameter_changed(AstroNorm) || parameter.check_parameter_changed(SpectralIndex);
    if (changed)
      recalculate(parameter);
    return changed;
  }

}  // namespace ana::ic
