#pragma once

// Double-float (df32) arithmetic: represent a value as a hi+lo pair of
// float32 so that error-free transformations (Dekker's TwoSum/TwoProd) carry
// roughly 44-48 bits of mantissa through +,-,*,exp -- without native fp64,
// which Apple Metal GPUs do not have.
//
// Every primitive below is built only from +,-,* and fma(), which are
// IEEE-754-conformant single-precision operations on both CPU float and
// Metal float when compiled with fastMathEnabled = NO (see GpuBackend.h's
// kNeumaierMetal comment for why that flag matters). That means this exact
// C++ logic, validated in Df64Test.cpp without a GPU, transcribes mechanically
// into the MSL kernel prelude: same operators, same rounding, same result.
//
// Precision floor: the reason plain FP32 (or FP32 + Neumaier-compensated
// summation) tops out at ~2.6e-7 relative error against a CPU FP64 reference
// (see GpuBackend.h) is that neither the per-event inputs (uploaded as a
// single float32 column) nor the per-event exp() are more precise than a
// single float32 mantissa -- no summation algorithm recovers that. df32
// widens both the inputs (hi+lo column upload) and the transcendental
// (df_exp below) themselves.

#include <cmath>

namespace ana::ic::df64 {

  struct Df32 {
    float hi;
    float lo;
  };

  // TwoSum (Knuth): s.hi + s.lo reproduces a+b to full double precision --
  // not an approximation, a theorem about binary floating point.
  inline Df32 two_sum(const float a, const float b) noexcept {
    const float s   = a + b;
    const float bb  = s - a;
    const float err = (a - (s - bb)) + (b - bb);
    return {s, err};
  }

  // TwoProd (Dekker, via fma): p.hi + p.lo reproduces a*b exactly. Requires a
  // true fused multiply-add (single rounding); Metal's fma() is IEEE-conformant.
  inline Df32 two_prod(const float a, const float b) noexcept {
    const float p   = a * b;
    const float err = std::fma(a, b, -p);
    return {p, err};
  }

  inline Df32 df_neg(const Df32 a) noexcept { return {-a.hi, -a.lo}; }

  inline Df32 df_add(const Df32 a, const Df32 b) noexcept {
    const Df32  s  = two_sum(a.hi, b.hi);
    const float lo = s.lo + a.lo + b.lo;
    return two_sum(s.hi, lo);
  }

  inline Df32 df_sub(const Df32 a, const Df32 b) noexcept { return df_add(a, df_neg(b)); }

  inline Df32 df_mul(const Df32 a, const Df32 b) noexcept {
    const Df32  p  = two_prod(a.hi, b.hi);
    const float lo = p.lo + a.hi * b.lo + a.lo * b.hi;
    return two_sum(p.hi, lo);
  }

  // Split a double into a df32 pair: hi is the nearest float, lo is the
  // (exactly representable in double, then rounded to float) residual. Used
  // host-side to prepare a df64 column upload from a double MC column.
  inline Df32 df_from_double(const double x) noexcept {
    const float  hi = static_cast<float>(x);
    const float  lo = static_cast<float>(x - static_cast<double>(hi));
    return {hi, lo};
  }

  inline double df_to_double(const Df32 a) noexcept {
    return static_cast<double>(a.hi) + static_cast<double>(a.lo);
  }

  // exp(x) for a df32 argument, via Cody-Waite range reduction (x = n*ln2 + r,
  // |r| <= ln2/2) and a 12-term Maclaurin series for exp(r) evaluated by
  // Horner in df64. Constants below are float32-pair splits of ln2, 1/ln2 and
  // 1/k! for k=0..12, computed offline to full double precision -- see the
  // MSL twin in GpuBackend.h (kDf64Metal) for the identical literals.
  inline Df32 df_exp(const Df32 x) noexcept {
    constexpr Df32 kLn2{0.6931471824645996f, -1.9046542121259336e-09f};
    constexpr float kInvLn2Hi = 1.4426950216293335f;

    constexpr Df32 kInvFact[13] = {
        {1.0f, 0.0f},
        {1.0f, 0.0f},
        {0.5f, 0.0f},
        {0.1666666716337204f, -4.967053879312289e-09f},
        {0.0416666679084301f, -1.2417634698280722e-09f},
        {0.008333333767950535f, -4.34617203337595e-10f},
        {0.0013888889225199819f, -3.3631094437103215e-11f},
        {0.00019841270113829523f, -2.725596874933456e-12f},
        {2.4801587642286904e-05f, -3.40699609366682e-13f},
        {2.7557318844628753e-06f, 3.793571224297229e-14f},
        {2.755731998149713e-07f, -7.575112209051195e-15f},
        {2.5052107943679403e-08f, 4.4176230446483665e-16f},
        {2.0876755879584152e-09f, 1.1082839809204342e-16f},
    };

    // n*ln2 must go through df_mul (TwoProd), not a component-wise n*hi /
    // n*lo: a plain float multiply of n by kLn2.hi rounds and silently
    // discards exactly the error term TwoProd exists to capture, which
    // otherwise reappears as an uncorrected ~1e-7-relative error in r.
    const float n   = std::round(x.hi * kInvLn2Hi);
    const Df32  r   = df_sub(x, df_mul(Df32{n, 0.0f}, kLn2));

    Df32 sum = kInvFact[12];
    for (int k = 11; k >= 0; --k) sum = df_add(df_mul(sum, r), kInvFact[k]);

    const float scale = std::exp2(n);
    return {sum.hi * scale, sum.lo * scale};
  }

}  // namespace ana::ic::df64
