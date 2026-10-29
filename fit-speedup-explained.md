# Making the IceCube fit faster: what was changed, and why

This document explains, step by step, everything that was changed on the branch
`feat/analytic-gradient` (6 commits on top of `59e619b`), the reasoning behind each
change, how each was validated, and what was found along the way. It assumes no
prior knowledge of how the fit works internally: Part 1 explains the machinery,
Part 2 the changes, Part 3 the evidence, Part 4 the findings, Part 5 how to use it.

All numbers come from **Asimov fits** of the ANTS tracks configuration
(`config_default.json` from `evaluation/ANTS/1D/asimov/1d-scan`, with the data path
removed), on the CUDA backend in FP64, on an RTX 2080 Ti. No fit was ever run on
real data.

---

## Summary

| | Before | After |
|---|---|---|
| Free fit (all 16 parameters floating) | ~24 s, ~2600 likelihood evaluations, several restarts | 2.9 s (Migrad, default) / 1.2 s (`--minimizer Newton`) |
| Points fitted per parameter in a crossing scan | 6–8 (+ a redundant free fit and node 0) | 4–6 |
| Time per scan point | 1.7–19 s | 0.1–1.3 s (Migrad) / 0.04–0.4 s (Newton) |
| `1d-crossing` for all 16 parameters | ~25 min (extrapolated from your run) | ~93 s (Migrad) / ~34 s (Newton) |
| Resulting intervals | — | agree with your earlier run to ≤ 0.5 % of their width |

What was done, in one line each:

- **A1–A5** (scan driver): stop fitting things that are already known or not needed.
- **B1**: the likelihood can now compute its own derivative exactly ("analytic gradient").
- **B2**: the likelihood also supplies an approximate second derivative ("Gauss-Newton
  Hessian"), and there is a new minimizer that exploits it ("Newton").
- **Bug fix**: a race condition in the CUDA backend that could corrupt GPU results.
- **Findings**: Migrad's convergence criterion can be badly over-optimistic; the
  Hesse errors are now more accurate (but still no replacement for profile scans of
  the signal parameters); SAY-type likelihoods do not recover the Asimov truth.

---

# Part 1 — How the fit works

## 1.1 What is being fitted

The analysis compares a **histogram of observed events** with a **predicted
histogram**. For the tracks sample the histogram has 50 × 33 = **1650 bins** in
(reconstructed log10 energy, cos zenith).

The prediction is built from **Monte Carlo (MC) events**: 16.2 million simulated
events that fall in the analysis range. Every MC event `i` sits in one bin and has a
**weight** `w_i` — how many real events it represents. The predicted number of
events in bin `b` is the sum of the weights of the MC events in that bin:

```
mu_b = sum over events i in bin b of w_i
```

The weight of an event depends on the **parameters** `theta` of the physics model.
Each event's weight is a sum of components:

```
w_i = astro_i + conv_i + prompt_i          (plus bin-level templates, see below)

astro_i  = AstroNorm_eff * baseline_i * (E_i / E_ref)^(2 - SpectralIndex)
conv_i   = ConvNorm  * [conv_base_i + CRGrad*(conv_alt_i - conv_base_i)]
                     * prod_k (1 + Barr_k * barr_slope_k,i / conv_base_i)
                     * (E_i / E_ref,conv)^(-DeltaGamma)
prompt_i = PromptNorm * [prompt_base_i + CRGrad*(...)] * (E_i / E_ref,prompt)^(-DeltaGamma)
```

On top of the per-event part come **bin-level** contributions that don't need the
events at all:

- the **muon template**: `MuonNorm * template_b`,
- the **detector systematics** (DOMEff, IceAbs, IceScat, HoleIceP0, HoleIceP1):
  a precomputed linear correction per bin, `sum_k (p_k - p_k,nominal) * gradient_k,b`.

The 16 free parameters are therefore of three kinds:

| Kind | Parameters | Cost of changing one |
|---|---|---|
| Normalisations | AstroNorm, ConvNorm, PromptNorm, MuonNorm | cheap in principle (rescale), but see 1.3 |
| Shape parameters | SpectralIndex, DeltaGamma, CRGrad, BarrH/W/Y/Z | expensive: every event weight changes → loop over 16 M events |
| Bin-level | DOMEff, IceAbs, IceScat, HoleIceP0, HoleIceP1, MuonNorm | cheap: 1650 bins |

Some parameters have **priors** (Barr, CRGrad, MuonNorm): a Gaussian penalty
`((x - x0)/sigma)^2` is added to the likelihood. These are the "pulls".

The 16 M-event loop runs on the **GPU** (CUDA, or Metal on a Mac). One full
evaluation of the likelihood takes about **4.4 ms** in FP64 on the 2080 Ti.

## 1.2 The likelihood

For each bin, the likelihood says how probable the observed count `k_b` is given
the prediction `mu_b`. The fit reports **-2 ln L** (smaller = better), summed over
bins, plus the pulls.

- **Poisson**: `-2 ln L = 2 * sum_b (mu_b - k_b ln mu_b + const)`. Assumes the
  prediction is exact.
- **SAY** (Argüelles, Schneider, Yuan, arXiv:1901.04645): the prediction itself is
  uncertain because it comes from a finite number of MC events. Its variance is
  estimated by the sum of squared weights,

  ```
  s_b = sum over events i in bin b of w_i^2       (the "MC variance", called ssq in the code)
  ```

  and the Poisson probability is replaced by a negative-binomial one with
  parameters `alpha = mu^2/s + a0`, `beta = mu/s`. `a0 = 1` is "L_Eff" (`SAY`),
  `a0 = 0` is "L_Mean" (`SAYMean`, your current choice). When `s → 0` this becomes
  Poisson again.

Each bin's contribution thus depends on **two** numbers: `mu_b` and `s_b`.
This matters for everything in Part 2.

**Asimov data**: instead of real data, `k_b` is set to the prediction at chosen
"true" parameter values (`AsimovValue`, or `StartValue`). A fit to Asimov data shows
the expected sensitivity. (`IC Asimov total events` in the log means this is what is
being fitted.)

## 1.3 Caching: why some parameter steps are cheap and others are not

The flux components remember the last parameters they were computed with
(`ParameterWrapper` tracks which parameters changed). If only a bin-level parameter
changed, the 16 M-event loop is skipped. If only a normalisation changed, the code
could in principle just rescale a cached histogram — but **with SAY it cannot**,
because `s_b = sum w_i^2` mixes the components *per event*
(`(a*astro_i + c*conv_i)^2` contains the cross term `2ac*astro_i*conv_i`).
So under SAY, every normalisation step also re-runs the event loop and the
`s_b` reduction. In practice almost every evaluation costs the full 4.4 ms.

## 1.4 What the minimizer does

The **minimizer** searches for the parameter values with the smallest -2 ln L. The
default is **Migrad** from ROOT's **Minuit2** library. It needs, at each step:

- the **gradient** `g` — the vector of first derivatives `d(-2lnL)/d theta_j`: which
  way is "downhill", and how steep;
- an estimate of the **Hessian** `H` — the matrix of second derivatives: how curved
  the valley is. Migrad builds this estimate up over the iterations (a
  "quasi-Newton" or "variable metric" method, BFGS-like updates).

A step is roughly `delta theta = -H^-1 g` (Newton's method), followed by a **line
search** along that direction.

**Before this branch, the likelihood could only return its value.** Minuit2 then
computes the gradient numerically by **finite differences**: change each free
parameter a little up and down and look at how the value changes. That costs
**2 evaluations per free parameter** (more with refinement cycles), i.e. ~32–35
evaluations per iteration for 16 parameters. Finite differences are also limited by
the rounding of the value: two values of ~8244 that differ only in the 10th digit
give noisy derivatives. Minuit reports this as *"Edm is limited by Machine
accuracy"*.

**Stopping criterion — the EDM.** Minuit stops when the *estimated distance to the
minimum* is small:

```
EDM = 1/2 * g^T * H^-1 * g        (the predicted remaining decrease of -2lnL)
converged  if  EDM < 0.002 * tolerance
```

With `--tolerance 1e-3`, Migrad requires EDM < 2e-6. Note that EDM is only an
*estimate* based on Migrad's own Hessian guess — see Part 4.2 for why that matters.

**Minuit strategy** (0, 1, 2) controls how much effort Minuit spends on second
derivatives: strategy 2 computes a full Hessian at the start and after
convergence (expensive with numerical derivatives, cheap with analytic ones).

## 1.5 Errors: Hesse

After convergence, the parameter **uncertainties** are estimated from the curvature
at the minimum: if -2 ln L were exactly a parabola, the covariance matrix would be
`V = 2 * H^-1` and the errors `sqrt(V_jj)`. Computing `H` is called **Hesse**.
Numerically it needs about n(n+1)/2 ≈ 140–190 extra evaluations.

Important: this is a **parabolic approximation**, valid only if the likelihood is
close to quadratic around the minimum (see Part 4.1).

**Bounds.** Some parameters have bounds (e.g. PromptNorm ≥ 0). Minuit handles them
by internally replacing the parameter with a transformed one (`x = sin(...)` or
`x = sqrt(...)` based) that can never leave the box. This transform becomes singular
when the parameter sits exactly on its bound — which is the case for PromptNorm in
this Asimov fit (its best fit is ~0).

## 1.6 Profile likelihood scans

A **profile likelihood scan** for parameter `x` fixes `x` at a value, lets all other
parameters float, minimises, and records -2 ln L. Repeating this for several `x`
values gives the *profile*. The 1σ interval is where the profile has risen by
**Δχ² = 1** above its minimum. Unlike Hesse, this captures asymmetric and
non-parabolic shapes.

The scan code in `programs/LLHFit/` places points on an integer **lattice**: node 0 is the
best-fit value, and neighbouring nodes are `unit = error/64` apart (so node ±64 is
±1 Hesse error). It does a **free fit** first (all parameters floating), saved as
`BestFit_<parameter>.json`, which defines node 0, the lattice spacing, and the
reference likelihood.

Two scan algorithms:

- **Walk** (`--scanMode 1d`): walks outward until Δχ² = 16 (4σ), mapping the whole
  profile.
- **Crossing search** (`--scanMode 1d-crossing`, what you used): only looks for the
  two points where Δχ² = 1. It works in `z = sqrt(Δχ²)`, which is linear in the
  parameter for a parabolic profile, so a straight-line ("secant") extrapolation
  through the fitted points lands near the crossing. It first probes at ±1 Hesse
  error (node ±64), then refines. The final crossing is interpolated in `z` between
  the two points that bracket the level.

Each scan point is a full fit of 15 parameters.

---

# Part 2 — What was changed, step by step

Each step below explains the problem, the idea, and then shows the code that
implements it ("In the code"). Snippets are taken from the branch and lightly
trimmed (comments shortened, `...` marks omitted lines).

## 2.0 Diagnosing where the time went

A single free Asimov fit with your settings (`--tolerance 1e-3`) showed:

- 4.4 ms per evaluation;
- Migrad's first pass: **74 iterations × ~35 evaluations** ≈ 2600 evaluations,
  almost all of them finite-difference gradients;
- then *"Edm is limited by machine accuracy"* and Migrad restarts — Minuit could not
  reach EDM < 2e-6 because the numerical gradient was too noisy.

Your scan directory showed further waste:

- `BestFit_AstroNorm`, `BestFit_ConvNorm`, … were **identical** 24 s fits, repeated
  for every parameter;
- `Crossing_<p>_0` re-fitted the best-fit point itself;
- fits that missed EDM 2e-6 (stopping at 1e-4 or so) were restarted up to 10 times
  (15–19 s each);
- the crossing search often spent 1–2 extra fits polishing a crossing that was
  already pinned down.

So there were two independent levers: **fewer fits** (the scan driver, A1–A5) and
**cheaper fits** (derivatives, B1–B2).

---

## 2.1 A1 — A separate tolerance for scan points (`--scanTolerance`)

**Idea.** The free fit's parameters and errors are reported, so it needs a tight
tolerance. A scan point only needs its -2 ln L accurate to well below the 0.02 in
Δχ² that the crossing search resolves. A separate option lets scan points use a
looser tolerance.

**Implementation.** `--scanTolerance` (`InputOptions`); `Fit::set_tolerance()`
stores the tolerance on the `Fit` so that Migrad restarts (which rebuild the
minimizer) keep it; `run_scan_batch()` sets it on every scan-point fit.

**In the code.** Every scan point is fitted inside `run_scan_batch()` (`ScanCommon.h`).
Before minimising, the fit gets the scan tolerance:

```cpp
// programs/LLHFit/ScanCommon.h, run_scan_batch()
ana::Fit fit(options, module, worker_index);
// A scan point only has to resolve the profile to well within the delta
// chi2 the scan works at, not to the free fit's --tolerance.
fit.set_tolerance(options->inputOptions().scan_tolerance());
fit.set_exact_errors(false);          // nothing reads a scan point's errors (see 2.7)
...
fit.minimize();
```

The tolerance lives on the `Fit`, not only on the minimizer, because a Migrad restart
throws the minimizer away and builds a new one:

```cpp
// libraries/likelihood/Fit.cpp
void Fit::set_tolerance(const double tolerance) {
  m_Tolerance = tolerance;                 // remembered for restarts ...
  m_Minimizer->SetTolerance(tolerance);    // ... and applied now
}
// in setup_minimizer(), which every restart calls:
m_Minimizer->SetTolerance(m_Tolerance);    // was: input_options.tolerance()
```

**Revised default (important).** Originally the default was 0.5 (EDM < 1e-3) for all
minimizers. Later testing (Part 4.2) showed that **Migrad's EDM can underestimate the
real distance to the minimum by ~300×**, so a "converged" Migrad scan point at
tolerance 0.5 could be 0.1–0.3 above its true minimum. The default is therefore now
**automatic**: `--tolerance` for Minuit2 (your previous behaviour — safe, and cheap
now that iterations are cheap) and 0.5 only for `--minimizer Newton`, whose stopping
was measured to be honest.

```cpp
// libraries/io/InputOptions.h
[[nodiscard]] double scan_tolerance() const noexcept {
  if (m_ScanTolerance > 0.0) return m_ScanTolerance;        // explicit --scanTolerance
  return m_Minimizer == "Newton" ? 0.5 : m_Tolerance;       // default (-1): pick by minimizer
}
```

## 2.2 A2 — Reuse the free fit across parameters

**Idea.** Every 1D scan without extra fixed parameters starts from the *same* free
fit. It was fitted again (24 s) for each parameter, only under a different file name.

**Implementation.** `reuse_sibling_free_fit()` in `ScanCommon.cpp`: before scanning
parameter P, if `BestFit_P.json` doesn't exist but `BestFit_Q.json` does for another
scanned parameter Q, it is copied to `BestFit_P.json`. The scan then reads it back
exactly as a *resumed* run would. Works for `1d`, `1d-crossing`, `1d-regular`, and for
single-parameter runs (`--scanParameter`), so separate invocations per parameter also
share it. N-1 scans (`nm1`) are not affected (their free fits differ).
**In the code.** Before each parameter's scan, the loop over parameters calls:

```cpp
// programs/LLHFit/ScanCommon.cpp
void reuse_sibling_free_fit(const std::string& prefix, const std::string& label,
                            const std::vector<std::string>& siblings,
                            const io::InputParameter& input_parameters, const std::string& format) {
  const std::filesystem::path target = prefix + label + result_extension(format);
  if (std::filesystem::exists(target))
    return;                                   // this parameter already has its free fit

  for (const auto& sibling : siblings) {      // every other scanned parameter
    if (sibling == label) continue;
    const std::string source = prefix + sibling;
    const auto stored = stored_fit(source, input_parameters, format);
    if (!stored || stored->parameters.empty())
      continue;                               // only reuse a usable (converged) fit
    std::filesystem::copy_file(source + result_extension(format), target);
    std::cout << "Reusing the free fit of " << sibling << " for " << label << '\n';
    return;
  }
}

// in perform_1d_crossing_all():
for (...each non-fixed parameter...) {
  reuse_sibling_free_fit("BestFit_", names[i], scanned, input_parameters, format);
  perform_1d_crossing(options, module, names[i], i);   // now finds BestFit_<name> on disk
}
```

`perform_1d_crossing()` itself was not changed: it already had a "resume" path that
reads `BestFit_<label>.json` if it exists instead of fitting. The copy simply makes
that path apply.

**A detail that needed fixing.** A fit read back from disk didn't carry its errors,
but the lattice spacing is `error/64`. The result readers (`stored_fit`, JSON and
protobuf) now also read the `error` fields, so a reused free fit produces exactly the
same lattice.

```cpp
// programs/LLHFit/ScanCommon.cpp, stored_fit(): read the errors next to the values
if (parameter.contains("error") && parameter["error"].is_number())
  errors[key] = parameter["error"].get<double>();
...
// bootstrap_seed_fit(), the "already stored" branch:
result.parameters = known->parameters;
result.errors     = known->errors;      // new: the lattice unit is errors[index] / 64
```

## 2.3 A3 — Don't refit node 0

**Idea.** Node 0 is the best-fit value of the scanned parameter. Fixing the
parameter there and fitting the rest just reproduces the free fit.

**Implementation.** `node_zero_from_free_fit()`: instead of fitting, copy
`BestFit_<label>.json` to `Crossing_<label>_0.json` (or `Output_<label>_0.json` for
the walk) and insert the free fit's likelihood into the profile. The file still
exists (for plots and for resuming), and the free fit's parameters seed the
neighbouring points as before.
**In the code.** The crossing search (and the walk) ask for a batch of nodes to be
fitted. Node 0 is now removed from the batch and filled from the free fit:

```cpp
// programs/LLHFit/ScanCommon.h
bool node_zero_from_free_fit(std::vector<int>& nodes, std::map<int, double>& profile,
                             const SeedFit& seed, const std::string& best_fit_name,
                             const std::string& node_file, const std::string& format,
                             bool warm_start, scanseed::Store<Key>& seeds) {
  const auto zero = std::ranges::find(nodes, 0);
  if (zero == nodes.end() || !std::isfinite(seed.llh))
    return false;                                        // node 0 not requested / no free fit
  ...
  if (!std::filesystem::exists(target))
    std::filesystem::copy_file(source, target);          // BestFit_X.json -> Crossing_X_0.json
  profile[0] = seed.llh;                                 // the profile's value at node 0
  if (warm_start && !seed.parameters.empty())
    seeds.store(0, seed.parameters);                     // neighbours start from it, as before
  nodes.erase(zero);                                     // don't fit it
  return true;
}

// perform_1d_crossing(): the callback that fits a batch of nodes
auto evaluate = [&](const std::vector<int>& nodes, scan1d::Profile& profile) {
  std::vector<int> remaining = nodes;
  node_zero_from_free_fit(remaining, profile, axis.seed, "BestFit_" + label,
                          "Crossing_" + label + "_0", format, warm_start, seeds);
  if (remaining.empty()) return;
  run_scan_batch(std::move(remaining), profile, ...);   // fit only the other nodes
};
```

## 2.4 A4 — Stop the crossing search on a narrow bracket

**Problem.** A side of the crossing search used to stop only when a point landed
within 0.02 of Δχ² = 1, or when two *adjacent* lattice nodes bracketed the level.
Your AstroNorm upper side went: node 64 (Δχ² = 7.55), node 23 (0.978 — missed the
tolerance by 0.002), node 25 (forced by a safety margin), node 24. Two of those fits
added nothing, because the final crossing is interpolated anyway.

**Change.** New setting `bracket_width = 4`: a side also stops once its bracket is
at most 4 lattice nodes wide (= 1/16 of the Hesse error). Over such a small range
`z` is essentially linear, so the interpolation is as good as another fit.
**In the code.** The search decides per side whether it is finished. `b` is the last
point below the level, `outer` the first point above it:

```cpp
// programs/LLHFit/CrossingScan1D.h, detail::finished()
if (side.outer) {                                     // the crossing is bracketed
  if (near_level(b) || near_level(*side.outer)        // a point within 0.02 of the level
      || std::abs(*side.outer - b) <= std::max(1, settings.bracket_width)) {  // was: <= 1
    result.crossing = estimate(profile, reference, side, target_z);  // interpolate in z
    result.status   = Status::converged;
    return result;
  }
}
```

`estimate()` is the interpolation in `z = sqrt(Δχ²)` between `b` and `outer`:

```cpp
// z rises linearly for a parabolic profile: solve z(x) = sqrt(level) on the segment
return b + (target_z - zb) * (*side.outer - b) / (zo - zb);
```

With your AstroNorm points: after node 23 (Δχ² 0.978) and node 25 (Δχ² 1.155) the
bracket [23, 25] is 2 nodes wide, so the side stops; node 24 is no longer fitted.

**Result.** AstroNorm: 5 fits instead of 8, same interval to 2e-4 of σ. A regression
test (`StopsOnANarrowBracket`) replays the measured AstroNorm profile shape.

## 2.5 A5 — Better starting conditions for scan points

**Idea.** A scan point's fit starts from the nearest already-fitted point (warm
start). Migrad also needs to know the *scale* on which each parameter moves. It used
the config's `StepWidth`, which was off by up to 10× (DOMEff: 0.02 configured, error
0.0026). The free fit's errors are the right scale.

**Implementation.** `Fit::set_step_sizes()` (kept across restarts); `run_scan_batch()`
passes the free fit's errors. This mainly helps the classic numerical path
(`--gradient false`).
**In the code.**

```cpp
// libraries/likelihood/Fit.cpp
void Fit::set_step_sizes(const std::span<const double> steps) {
  m_StepSizes.assign(steps.begin(), steps.end());           // kept for restarts
  for (std::size_t i = 0; i < m_StepSizes.size(); ++i)
    if (std::isfinite(m_StepSizes[i]) && m_StepSizes[i] > 0.0)
      m_Minimizer->SetVariableStepSize(i, m_StepSizes[i]);
}
// setup_minimizer(): the step of each parameter
const double step = i < m_StepSizes.size() && m_StepSizes[i] > 0.0 ? m_StepSizes[i]
                                                                   : parameters[i].uncertainty();

// programs/LLHFit/ScanCommon.h, run_scan_batch():
if (!step_sizes.empty())
  fit.set_step_sizes(step_sizes);     // the callers pass seed.errors (the free fit's errors)
```

The other half of A5 — seeding Migrad with a full covariance matrix — is achieved
differently: with the analytic Hessian (B2) every fit starts from an exact metric.
ROOT's `SetCovariance()` was deliberately not used: its conversion to internal
coordinates mixes up parameter indices when fixed parameters sit between free ones
(MuonGunNorm and VetoThreshold do, here), which would silently seed the wrong
values.

---

## 2.6 B1 — The analytic gradient

This is the core change.

### Why

With finite differences, each Migrad iteration costs ~35 evaluations (2 per free
parameter, plus the line search) and is limited by rounding noise. If the likelihood
computes `g` exactly itself, one gradient costs about as much as one or two
evaluations, and it is exact. Migrad then needs ~3 evaluations per iteration instead
of ~35, and it no longer hits the machine-accuracy limit.

### The chain rule, in two steps

-2 ln L depends on the parameters only through the per-bin numbers `mu_b` and `s_b`
(and, for galactic templates, a per-bin addition). So:

```
d(-2lnL)/d theta_j  =  sum_b [ d(-2lnL_b)/d mu_b * d mu_b/d theta_j
                              + d(-2lnL_b)/d s_b  * d s_b/d theta_j ]
                     + d(pulls)/d theta_j
```

That splits the work into two independent pieces:

1. **How mu_b and s_b change with each parameter** — needs the events (GPU), done by
   the new `FluxGradient` component, plus the bin-level components.
2. **How each bin's term changes with mu_b and s_b** — a formula per bin, done on the
   CPU (`BinTermDerivatives.h`), 1650 bins, negligible cost.

### Piece 1: `FluxGradient` — one sweep over the events
First, what has to come out of the sweep. The chain rule needs, per bin and per
parameter, `d mu_b/d theta` and `d s_b/d theta`. Writing the event weight as
`w_i = a_eff·wa_i + c·cw_i + p·pw_i` (norms times norm-free components), everything
reduces to 29 per-bin sums. They are named in an enum, so the CPU loop, both GPU
kernels and the code reading the table use the same indices:

```cpp
// libraries/likelihood/IceCube/FluxGradient.h
enum Quantity : int {
  SumAstro = 0, DAstro0, DAstro1, DAstro2,   // sum wa_i, sum d wa_i / d(astro shape params)
  SumConv, SumPrompt,                        // sum cw_i, sum pw_i
  DConvCR, DConvDG, DConvB0, DConvB1, DConvB2, DConvB3, DConvVeto,   // sum d cw_i / d theta
  DPromptCR, DPromptDG, DPromptVeto,                                  // sum d pw_i / d theta
  SsqAstroNorm, SsqAstro0, SsqAstro1, SsqAstro2, SsqConvNorm, SsqPromptNorm,
  SsqCR, SsqDG, SsqB0, SsqB1, SsqB2, SsqB3, SsqVeto,                  // sum w_i * d w_i / d theta
  kQuantities                                                         // = 29
};
```

The derivative of each event weight is a simple expression, because the weights are
products of simple factors. Examples:

```
d astro_i / d SpectralIndex = -ln(E_i/E_ref) * astro_i
d conv_i  / d DeltaGamma    = -ln(E_i/E_ref,conv) * conv_i
d conv_i  / d CRGrad        = (conv_alt_i - conv_base_i) * (all other factors)
d conv_i  / d Barr_k        = conv_i * slope_k,i/base_i / (1 + Barr_k * slope_k,i/base_i)
d astro_i / d AstroNorm     = astro_i / AstroNorm   (computed without dividing)
```

and for the SAY variance, since `s_b = sum_i w_i^2`:

```
d s_b / d theta = 2 * sum_i w_i * d w_i / d theta
```
**In the code — the per-event part.** This is the body of the GPU loop (the CPU
version in `astro_terms()` / `atmo_terms()` / `accumulate()` does the same in C++).
`real` is `double` on CUDA FP64 and `float` on CUDA FP32 and Metal; the same text is
compiled for all three:

```c
// FluxGradient.cpp, kEventBody (runs over the events i of one chunk)
const real lnE = log_e_true[i];

// astrophysical, single power law: wa = baseline * (E/E_ref)^exponent
const real lr = lnE - p.log_eref;
wa  = astro_base[i] * REXP(p.exponent * lr);
ka0 = -lr;                         // d wa / d SpectralIndex = ka0 * wa

// conventional
const real lin  = cb + p.cr * (alt - cb);            // CRGrad interpolation
const real inv  = (real)1 / cb;                      // one division instead of four
const real x0 = barr0[i] * inv, ...;                 // Barr slope / baseline
const real B0 = (real)1 + p.barr0 * x0, ...;         // Barr factors
real ev = REXP(-p.dg * (lnE - p.log_eref_conv));     // DeltaGamma tilt
const real rest = B0 * B1 * B2 * B3 * ev;
cw   = lin * rest;                                   // the weight itself
dccr = (alt - cb) * rest;                            // d cw / d CRGrad
dcdg = -(lnE - p.log_eref_conv) * cw;                // d cw / d DeltaGamma
dcb0 = lin * ev * x0 * B1 * B2 * B3;                 // d cw / d BarrH (product rule)
...
// accumulate the per-bin sums
const real w = p.a_eff * wa + p.conv_norm * cw + p.prompt_norm * pw;   // total weight
acc[Q_SumConv]  += cw;
acc[Q_DConvCR]  += dccr;
acc[Q_SsqCR]    += w * (p.conv_norm * dccr + p.prompt_norm * dpcr);   // w * dw/dCRGrad
...
```

After the loop, each GPU block (256 threads, one "chunk" of up to 8192 events) sums
its threads' 29 accumulators and writes 29 partial sums; a second small kernel
(`bin_gather_all`) adds up the partials of each bin. On CUDA the block sum uses warp
shuffles:

```c
for (int q = 0; q < NQ; ++q) {
  real v = acc[q];
  for (int offset = 16; offset > 0; offset >>= 1)
    v += __shfl_down_sync(0xffffffffu, v, offset);   // sum within a warp of 32 threads
  if (lane == 0) sdata[warp * NQ + q] = v;           // one value per warp
}
__syncthreads();
if (tid < NQ) {                                       // thread q adds up quantity q
  real v = 0;
  for (unsigned int k = 0; k < (nthreads >> 5); ++k) v += sdata[k * NQ + tid];
  partial[tid * n_chunks + chunk] = v;
}
```

`FluxGradient` computes, in **one loop over the events**, 29 per-bin sums:

- the norm-free sums `sum astro_i`, `sum conv_i`, `sum prompt_i` (derivatives with
  respect to the norms),
- their derivatives with respect to every shape parameter,
- `sum_i w_i * dw_i/dtheta` for every flux parameter (for `s_b`).

The loop reads the same MC columns as one likelihood evaluation, so its memory
traffic is similar. It exists in three implementations:

- **CPU** — the reference, also used for forward-folded samples (where an event
  contributes to several bins with a "response fraction" f; the mu-terms then get a
  factor f and the s-terms f²);
- **CUDA** — in the backend's precision (FP64 for you); the per-block reduction uses
  warp shuffles;
- **Metal** — FP32. Derivatives only steer the minimizer, so FP32 is adequate there
  even though the likelihood value itself needs more precision.

All four astrophysical models (single power law, broken power law, log-parabola,
cutoff) and the veto reweighting are covered. A new single-dispatch gather kernel
(`GpuBinReduce::gather_all`) sums the 29 quantities per bin in one launch. The GPU
setup is lazy (first use), so `--gradient false` costs nothing extra.

In FP64 on the 2080 Ti one gradient sweep takes ~12 ms (vs 4.4 ms for an evaluation):
consumer GPUs run FP64 at 1/32 speed, and the gradient does about three times the
arithmetic per event. On HPC GPUs (A100/H100, full FP64 rate) the ratio is better.

The bin-level components contribute their derivatives directly: the muon template
(`d mu = template`, `d s = 2 * norm * sigma²`), the detector systematics (linear in
mu, quadratic in s via the gradient errors and covariances), and the galactic
templates (analysis binning, no variance).

### Piece 2: derivatives of the bin terms

For Poisson: `d(-2lnL_b)/d mu_b = 2 (1 - k_b/mu_b)`.

For SAY the bin term is

```
l_b = alpha ln beta + lgamma(k+alpha) - lgamma(k+1) - (k+alpha) ln(1+beta) - lgamma(alpha)
alpha = mu^2/s + a0,   beta = mu/s
```

Its derivatives with respect to `alpha` and `beta` involve the **digamma** function
ψ (derivative of lgamma) and, for second derivatives, the **trigamma** ψ₁. These are
implemented in `Polygamma.h`. A subtlety: in bins with many MC events, `alpha` is
huge (10⁶ and more), and the formula needs `ψ(k+alpha) − ψ(alpha)`, two nearly equal
numbers. Subtracting them directly would lose most digits, so the difference is
computed from the asymptotic series term by term (`digamma_difference`), which stays
accurate.
**In the code — the SAY bin term.** Derivatives with respect to `alpha` and `beta`
first, then converted to `mu` and `s` with the chain rule
(`alpha = mu²/s + a0`, `beta = mu/s`):

```cpp
// libraries/likelihood/IceCube/BinTermDerivatives.h, say_derivatives()
const double alpha = mu * mu / s + alpha_offset;
const double beta  = mu / s;
const double opb   = 1.0 + beta;

// dl/dalpha = -n log1p(1/beta) + sum over bins of [psi(k + alpha) - psi(alpha)]
double l_a = -n * std::log1p(1.0 / beta);
each_count([&](double k) { l_a += polygamma::digamma_difference(k, alpha); });

// dl/dbeta = n alpha / beta - (K + n alpha)/(1 + beta), rewritten without cancellation
const double l_b = n * alpha / (beta * opb) - k_total / opb;

// chain rule to (mu, s)
d.mu = l_a * (2.0 * mu / s)       + l_b * (1.0 / s);
d.s  = l_a * (-mu * mu / (s * s)) + l_b * (-mu / (s * s));
```

The cancellation-free digamma difference:

```cpp
// libraries/likelihood/IceCube/Polygamma.h
inline double digamma_difference(const double k, const double a) noexcept {
  if (a < 10.0)
    return digamma(a + k) - digamma(a);          // small a: plain difference is fine
  const double b = a + k;
  // psi(x) ~ ln x - 1/(2x) - tail(x): difference the terms analytically
  return std::log1p(k / a) + 0.5 * k / (a * b) - (digamma_tail(b) - digamma_tail(a));
}
```

The code also reproduces exactly the special cases the likelihood itself takes:
prediction ≤ 0 (no derivative), variance ≤ 0 (falls back to Poisson), variance > mu²
(clipped to mu² — then the term depends on mu through the clip). If the gradient
didn't follow the same branches as the value, the minimizer would be steered by the
derivative of a different function.
The branches, mirroring `say_llh()`:

```cpp
// BinTermDerivatives.h, say_term()
if (value <= 0.0) return {};                                   // prediction clamped at 0: flat
if (s <= 0.0)     return poisson_derivatives(mu, n, k_total);  // no MC variance: Poisson
if (s < mu * mu)  return say_derivatives(mu, s, ...);          // the ordinary case
// s >= mu^2: the likelihood uses s = mu^2, so the term depends on mu through the clip
const TermDerivatives c = say_derivatives(mu, mu * mu, ...);
d.mu = c.mu + 2.0 * mu * c.s;                                  // d/dmu of l(mu, mu^2)
```

Bins sharing one MC bin across RA (3D binnings) are handled as a group, exactly as
`say_llh()` does.
**In the code — putting the two pieces together.** Per sample, a list of "columns"
holds, for each parameter, `d mu_b/d theta` and `d s_b/d theta` for every bin. Each
bin then contributes its term's derivatives times the columns:

```cpp
// libraries/likelihood/IceCube/SampleLikelihoodGradient.cpp, accumulate_gradient()
static_cast<void>(partial_llh(parameter));          // bring mu_b, s_b to this point
m_FluxGradient->compute(parameter, m_GradTable);    // the 29 sums from the GPU

// e.g. the CRGrad column: d mu/dCR = c*sum dcw/dCR + p*sum dpw/dCR,  d s/dCR = 2*SsqCR
GradientColumn& cr = column(CRGrad);
add(cr, c_norm, Q::DConvCR);
add(cr, p_norm, Q::DPromptCR);
add_ssq(cr, Q::SsqCR);
...                                                 // all flux params, template, detector

for (std::size_t b = 0; b < n_mc; ++b) {
  TermDerivatives d = say_term(m_McTotal[b] / n_ra, m_McSsq[b] / (n_ra * n_ra), ...);
  const double g_t = -2.0 * d.mu / n_ra;            // d(-2lnL)/d mu_b  (MC binning)
  const double g_s = -2.0 * d.s / (n_ra * n_ra);    // d(-2lnL)/d s_b
  for (std::size_t c = 0; c < n_columns; ++c)
    gradient[params[c]] += g_t * m_GradColumns[c].dmu[b] + g_s * m_GradColumns[c].dssq[b];
}
```

`ICLikelihood` sums the samples and adds the pulls:

```cpp
// libraries/likelihood/IceCube/ICLikelihood.cpp, evaluate_derivatives()
if (same_point && (m_HaveHessian || !want_hessian))
  return;                                      // Minuit often asks twice at one point
m_Parameter.reset_parameter(parameter);
for (const auto& sample : m_Samples)
  sample->accumulate_gradient(m_Parameter, m_Gradient, want_hessian ? m_Hessian.data() : nullptr);
for (const auto& [idx, cv, sigma] : m_Pulls)
  m_Gradient[idx] += 2.0 * (m_Parameter[idx] - cv) / (sigma * sigma);   // d/dx ((x-cv)/sigma)^2
```

### How Minuit2 gets it

`Likelihood` (the base class of all experiments) gained `has_gradient()`,
`calculate_gradient()` and `calculate_hessian()` with default "not available".
`ICLikelihood` implements them and caches the last result per parameter point
(Minuit2 asks for the gradient and the Hessian at the same point). `Fit` wraps the
likelihood in a ROOT `GradFunctor` when `--gradient true` (the default) and the
likelihood has a gradient; other experiments (DoubleChooz, LinearRegression) are
unaffected.

```cpp
// libraries/likelihood/Fit.cpp, Fit::Fit()
auto value = [this, likelihood](const double* x) {
  ++m_NCalls;
  return likelihood->calculate_likelihood(x);
};
m_UseGradient = options.analytic_gradient() && likelihood->has_gradient();
if (m_UseGradient) {
  // value + gradient: Minuit2 uses the analytic gradient instead of finite differences
  m_Functor = std::make_shared<ROOT::Math::GradFunctor>(value, n_parameter,
      [this, likelihood](const double* x, double* gradient) {
        ++m_NGradCalls;
        likelihood->calculate_gradient(x, gradient);
      });
} else {
  m_Functor = std::make_shared<ROOT::Math::Functor>(value, n_parameter);   // as before
}
```

### How it was validated

- **Unit tests** (`GradientTests.cpp`): the analytic gradient is compared with
  central finite differences of the actual `partial_llh()` for every likelihood
  type × every astro model, with templates and detector systematics, with the veto,
  with RA bins and galactic templates, and for a forward-folded sample. Agreement:
  2e-5 relative.
- **Branch tests**: every branch of the SAY bin term (ordinary, clipped, Poisson
  fallback, large alpha, empty bins, non-integer counts) against finite differences.
- **GPU vs CPU**: CUDA FP64 agrees with the CPU to 1e-9, CUDA FP32 to 2e-4
  (Metal is in the same test but could not run on Linux).
- **Mutation testing**: deliberate small errors were inserted into the gradient code
  (a wrong factor in the variance chain, in a Barr derivative, in the GPU kernel
  only, a missing covariance term, a wrong clip branch); each was caught by the tests.
- **On the real Asimov**: the new `GradientCheck` program agrees with finite
  differences to ≤ 6e-6 for all 16 parameters.

---

## 2.7 B2 — Second derivatives: the Gauss-Newton Hessian

### Why

Migrad builds its Hessian estimate from scratch over many iterations, starting from
a crude diagonal guess. With the derivative machinery from B1, a good Hessian
approximation is almost free.

### What the Gauss-Newton Hessian is

Differentiating the chain rule above once more gives

```
H_jk = sum_b  J_b,j^T * W_b * J_b,k          (term 1)
     + sum_b  d(-2lnL_b)/d(mu,s) * d²(mu,s)/d theta_j d theta_k     (term 2)
```

where `J_b = (d mu_b/d theta, d s_b/d theta)` are exactly the per-bin derivatives
from B1, and `W_b` is the 2×2 matrix of second derivatives of the bin term with
respect to `(mu_b, s_b)`. The **Gauss-Newton** approximation keeps term 1 and drops
term 2 (which would need second derivatives of every event weight with respect to
every pair of parameters — far more GPU sums).

Term 2 is weighted by `d(-2lnL_b)/d mu_b`, which is zero where the prediction matches
the data. So Gauss-Newton is exact there and a good approximation nearby. Each bin's
2×2 block `W_b` is clipped to be non-negative definite, so the whole matrix always
describes a bowl, never a saddle — which is what a minimizer needs.
**In the code.** Inside the same bin loop as the gradient, if a Hessian is requested:

```cpp
// SampleLikelihoodGradient.cpp
double wtt = -2.0 * d.mumu / (n_ra * n_ra);              // W_b, the bin's 2x2 block
double wts = -2.0 * d.mus  / (n_ra * ssq_divisor);
double wss = -2.0 * d.ss   / (ssq_divisor * ssq_divisor);
clip_to_psd(wtt, wts, wss, m_McTotal[b]);                // drop negative curvature
// H += J_b^T W_b J_b, with J_b = (dmu_b, ds_b) per parameter column
add_outer(params, jm.data(), js.data(), wtt, wts, wss);
```

### How Minuit2 uses it — and two technical problems

`Fit` registers the Hessian with Minuit2. Minuit2 then uses it for the starting
metric of Migrad and for Hesse.

- **Strategy 2 is needed.** With strategy 1, Minuit2 uses only the *diagonal* of a
  user Hessian to start, and on this problem Migrad then did not converge. With
  strategy 2 it uses the whole matrix (one cheap call). So `--hessian gn` raises the
  strategy to 2.
- **A ROOT/C++ incompatibility.** ROOT's `Minimizer::SetHessianFunction()` takes a
  function of a `std::span`. Your ROOT library was compiled in C++17 mode, where
  ROOT uses its own replacement `span`; this project compiles as C++23, where
  `std::span` is the real one. The same declaration therefore means two different
  types on the two sides of the library boundary, and the first call crashed.
  The workaround: `Fit` constructs Minuit2 through a small subclass that attaches the
  Hessian directly to Minuit2's internal function adapter, whose setter uses
  `std::vector` (identical on both sides). The subclass also overrides the
  span-typed functions so that it links. The code is guarded by ROOT version (before
  ROOT 6.34 these functions had different signatures), so it builds on older ROOT
  as well — relevant for the MacBook. ROOT 6.38 also merged the template
  `FCNGradAdapter<T>` into the non-template `FCNAdapter` (same `std::vector`-based
  `SetHessianFunction`); `Fit.cpp` picks whichever header exists with
  `__has_include`, so it builds against ROOT master too (needed on macOS 27).

  ```cpp
  // libraries/likelihood/Fit.cpp
  class Minuit2 final : public ROOT::Minuit2::Minuit2Minimizer {
   public:
    using ROOT::Minuit2::Minuit2Minimizer::Minuit2Minimizer;

    template <class Hessian>
    void set_hessian(Hessian hessian) {
  #if __has_include(<Minuit2/FCNGradAdapter.h>)                           // ROOT < 6.38
      using Adapter = ROOT::Minuit2::FCNGradAdapter<ROOT::Math::IMultiGradFunction>;
  #else
      using Adapter = ROOT::Minuit2::FCNAdapter;
  #endif
      auto* fcn = dynamic_cast<const Adapter*>(GetFCN());                  // Minuit2's wrapper
      const_cast<Adapter*>(fcn)->SetHessianFunction(std::move(hessian));   // std::vector-based
    }
  #if ROOT_VERSION_CODE >= ROOT_VERSION(6, 34, 0)
    void SetHessianFunction(std::function<bool(std::span<const double>, double*)> h) override {...}
    bool SetCovariance(std::span<const double>, unsigned int) override { return unsupported(); }
    ...
  #endif
  };

  // setup_minimizer():
  minuit->set_hessian([hessian](const std::vector<double>& x, double* out) { return hessian(x.data(), out); });
  m_Minimizer->SetStrategy(m_UseHessian ? std::max(2, strategy) : strategy);   // strategy 2 with GN
  ```

### Result

The free Asimov fit: **2.55 s** (166 evaluations + 67 gradients) instead of 24 s,
at the same minimum (LLH equal to 1e-6, parameters to 1e-4 σ).

### The errors problem, and its solution: exact Hesse

The Gauss-Newton Hessian is a good *metric for minimising* but gave **wrong errors**:
AstroNorm 0.47 instead of 1.17, SpectralIndex 0.07 instead of 0.17, detector
parameters ×1.35. The reasons: (a) the dropped term 2 is large here — this Asimov fit
does not recover its truth (AstroNorm 2.72 vs 1.77, see Part 4.3), so the
"prediction matches data" condition does not hold; (b) the SAY variance `s` is
quadratic in the norms and the detector parameters, and its second derivatives are
exactly what Gauss-Newton drops.

**Solution.** After Migrad has converged (on the cheap GN metric), `Fit` switches the
Hessian callback to the **exact Hessian**, computed by central finite differences
*of the analytic gradient* (2 gradients per free parameter, ~0.5 s), and calls Hesse
once more. Differencing an exact gradient is far more accurate than Minuit's
second differences of values. For parameters near a bound, the exact Hessian is
corrected by the extra curvature term that Minuit's internal bound transform adds
(`f' * x''(int) / x'(int)²`), so the errors keep the meaning they always had in
Minuit. Scan points skip this step (nobody reads their errors).
**In the code.**

```cpp
// Fit::minimize(), after Migrad (and any restarts):
if (m_UseHessian && m_ExactErrors) {
  m_ExactHessianMode = true;       // the Hessian callback now returns the exact Hessian
  m_Minimizer->Hesse();            // Minuit2 computes the errors from it
  m_ExactHessianMode = false;
}

// Fit::exact_hessian(): central differences of the analytic gradient
for (each free parameter i) {
  const double h = 1.0e-3 * error_i;                 // a thousandth of its current error
  point[i] = x[i] + h;  m_Likelihood->calculate_gradient(point.data(), up.data());
  point[i] = x[i] - h;  m_Likelihood->calculate_gradient(point.data(), down.data());
  for (j) hessian[i * n + j] = (up[j] - down[j]) / (2 * h);   // column i of H
}
// + symmetrise, + Minuit's bound-transform term g_i * x''(int) / x'(int)^2 on the diagonal
```

**Validation against the profile likelihood** (the reference for "correct"):

| Parameter | Profile crossings −/+ (half-width) | New exact Hesse | Old Minuit numerical Hesse |
|---|---|---|---|
| BarrZ | 0.0851 / 0.0962 (0.0906) | **0.0903** | 0.1085 |
| CRGrad | 0.4816 / 0.4384 (0.460) | **0.454** | 0.486 |
| ConvNorm | 0.1027 / 0.1182 (0.1105) | **0.1102** | 0.1151 |
| DOMEff | 0.00257 / 0.00256 | 0.00257 | 0.00257 |

Why the old numerical Hesse was off for the atmospheric parameters: they are
strongly correlated (BarrZ's error with everything else fixed is ~0.0017, its
marginal error ~0.09 — 50× larger). Inverting such a matrix amplifies small errors in
its elements by ~50² ≈ 2500. Minuit's second differences of likelihood values have
element errors of order 1e-5–1e-4, enough to shift the marginal errors by 5–20 %.

---

## 2.8 B2, second part — the Newton minimizer (`--minimizer Newton`)

### Idea

With an exact gradient and a good Hessian at every point, the textbook **Newton
method** (step `-H^-1 g`, recomputing `H` at each point) needs far fewer iterations
than Migrad's quasi-Newton updates.

### What it does (`NewtonMinimizer.cpp`)

Each iteration:

1. Build the **free set**: parameters that are not fixed and not "held" on a bound.
2. Solve `(H + λ diag(H)) d = -g` over the free set. `λ` is the **Levenberg-Marquardt
   damping**: λ = 0 is a pure Newton step; larger λ gives a shorter, more cautious
   step closer to steepest descent.
3. Project `x + d` back into the parameter box (clip at the bounds).
4. Compare the actual decrease of -2 ln L with the decrease the quadratic model
   predicted (`rho = actual/predicted`). Accept the step if it decreased reasonably
   (`rho > 1e-4`); decrease λ after acceptable steps, increase it after rejected ones.
5. Stop when the EDM (same definition and same tolerance meaning as Migrad, so
   `--tolerance` means the same thing) is below `0.002 × tolerance`.

**Bounds without a transform.** Minuit maps bounded parameters through a nonlinear
transform that becomes singular at the bound. Newton instead works in the
parameters' own coordinates and uses an **active set**: a parameter on its bound is
held there while the gradient — or the Newton step — pushes it outward (the
"binding set" rule). PromptNorm sitting at 0 is then an ordinary situation. Without
the "Newton step pushes outward" part, PromptNorm was released and re-clipped over and
over and steps kept being rejected; with it, the fit stopped "chattering".

**Errors.** `Hesse()` inverts the exact Hessian (from `Fit`, as in 2.7). Parameters
held on a bound are treated as fixed for the others' errors.

It is a `ROOT::Math::BasicMinimizer`, so the rest of the code (scans, result writers)
sees an ordinary ROOT minimizer.
**In the code — one iteration** (`NewtonMinimizer::Minimize()`, trimmed):

```cpp
build();                                   // free set, H and g restricted to it, Newton step
edm = -0.5 * gf.dot(newton);               // = 1/2 g^T H^-1 g
if (edm < edm_target) { converged = true; break; }

for (int attempt = 0; attempt < 40; ++attempt) {
  Matrix damped = hf;                                    // H + lambda * diag(H)
  for (a) damped(a, a) += lambda * std::max(hf(a, a), 1e-12);
  solve_floored(damped, -gf, step);                      // solve for the step

  std::vector<double> trial = x;                         // take it, but stay inside the box
  for (a) trial[free[a]] = std::clamp(x[free[a]] + step(a), lower, upper);

  const double predicted = -(gf.dot(dp) + 0.5 * dp.dot(hf * dp));   // model decrease
  const double ft        = function(trial);                         // actual value
  const double rho       = (fx - ft) / predicted;                   // how good was the model?

  if (rho > 1e-4) {                                      // accept
    x = trial; fx = ft;
    hessian(x, h); gradient(x, g);                       // new H and g (one GPU sweep)
    lambda = rho < 0.25 ? std::max(2 * lambda, 1e-3) : lambda / 3;   // trust the model less/more
    break;
  }
  lambda = (lambda == 0) ? 1e-3 : growth * lambda;       // reject: shorter, more cautious step
}
```

The binding-set rule inside `build()`:

```cpp
// a free parameter on its bound whose Newton step points out of the box is held, and
// the step is recomputed without it, until no such parameter is left
if ((x[i] <= lower_i && newton(a) < 0.0) || (x[i] >= upper_i && newton(a) > 0.0)) {
  hold[i] = 1;
  changed = true;
}
```

### Things tried and dropped

- A **quasi-Newton correction** on top of Gauss-Newton (PSB update, the NL2SOL idea)
  made the matrix indefinite and convergence worse; removed.
- A **rank-one curvature correction** along the last step: no gain, broke a test;
  removed.
- **Nielsen's damping rule**: slightly worse than the simple one; reverted.

### Result

Free fit: **1.2 s** (30 iterations, 32 evaluations + ~30 gradients, plus ~33
gradients for the exact Hesse). Scan points: 0.04–0.4 s. Unit tests: correlated
quadratic (exact in one step, exact errors), Rosenbrock, minimum outside the box,
fixed parameter, and agreement with Migrad on a Poisson toy fit.

---

## 2.9 Bug fix — a race condition in the CUDA backend

**Symptom.** About one in eight Newton crossing scans had one scan point that failed
completely (every trial step rejected, even with tiny steps), and once that moved
AstroNorm's lower crossing to −0.41 instead of −1.19. The failing point always
started from the same likelihood value as in the good runs, but with a different
EDM — i.e. the **first gradient of a fresh likelihood** was occasionally wrong.

**Cause.** When the CUDA backend allocates an output buffer it zero-fills it with
`cuMemsetD8`, which runs *asynchronously* on CUDA's default ("NULL") stream. The
kernels run on the sample's own stream, created as `CU_STREAM_NON_BLOCKING`, which by
definition does **not** wait for work on the NULL stream. A kernel launched right
after an allocation could therefore write its results first and then have them
zeroed by the late memset.

The existing flux components never showed this: their buffers are allocated in the
constructor and followed by synchronous column uploads, which happen to wait for the
NULL stream. `FluxGradient` allocates on first use and launches immediately, which
exposed the bug.

**Fix.** The zero-fill now uses `cuMemsetD8Async` on the session's own stream, so it
is ordered before every later kernel. After the fix: 12 of 12 repeated scans
identical (before: 2 failures in 16). `compute-sanitizer` (memcheck, initcheck,
racecheck) is clean. This was a latent bug in the existing backend, so the fix also
protects the value path. Metal is not affected (its buffers are zero-filled by the
CPU synchronously).

```cpp
// libraries/likelihood/IceCube/CudaBackend.cpp, CudaSession::alloc_output()
cu_check(cuMemAlloc(&row.dptr, n * row.elem_bytes), "cuMemAlloc(output)");
// before:  cuMemsetD8(row.dptr, 0, n * row.elem_bytes);              -> NULL stream, unordered
cu_check(cuMemsetD8Async(row.dptr, 0, n * row.elem_bytes, s->stream),  // the session's stream
         "cuMemsetD8Async(output)");
```

---

## 2.10 `GradientCheck` — a validation tool

`programs/GradientCheck` loads a configuration exactly like `LLHFit` and compares the
analytic gradient with finite differences of the likelihood (at two step sizes, so
the finite-difference noise itself is visible), optionally also the Hessian diagonal
(`--hessianCheck`, Gauss-Newton vs exact). Use it to validate a new backend (Metal on
the MacBook) or a new configuration. See Part 5.
The core of it:

```cpp
// programs/GradientCheck/GradientCheck.cpp
likelihood->calculate_gradient(x.data(), gradient.data());   // analytic
auto central = [&](std::size_t i, double h) {                // numerical
  p[i] = x[i] + h; const double up   = likelihood->calculate_likelihood(p.data());
  p[i] = x[i] - h; const double down = likelihood->calculate_likelihood(p.data());
  return (up - down) / (2.0 * h);
};
const double fd     = central(i, h);
const double fd2    = central(i, h / 2);
const double richer = (4.0 * fd2 - fd) / 3.0;               // Richardson extrapolation
// rel.diff = |analytic - richer| / scale;   fd.noise = |fd - fd2| / scale
```

---

# Part 3 — Validation evidence

## 3.1 Free fit (Asimov, all 16 parameters)

| | Old (Migrad, numerical) | Migrad + gradient + GN | Newton |
|---|---|---|---|
| Time | ~24 s (with restarts) | 2.9 s | 1.2 s |
| LLH | 8244.248486 | 8244.248487 | 8244.248487 |
| Parameters | — | same to < 1e-3 σ | same to < 1e-3 σ |

## 3.2 Full crossing scan (16 parameters, Asimov)

Intervals (−/+) from your original run, the new Migrad default, and Newton:

| Parameter | Old run | Migrad + GN (default) | Newton |
|---|---|---|---|
| AstroNorm | 1.19479 / 0.42581 | 1.19468 / 0.42578 | 1.19447 / 0.42585 |
| SpectralIndex | 0.19911 / 0.07096 | 0.19913 / 0.07095 | 0.19909 / 0.07095 |
| ConvNorm | 0.10269 / 0.11824 | 0.10270 / 0.11823 | 0.10268 / 0.11824 |
| PromptNorm | at bound / 2.58942 | at bound / 2.59789 | at bound / 2.59866 |
| BarrH | 0.13618 / 0.13763 | 0.13618 / 0.13766 | 0.13616 / 0.13762 |
| BarrW | 0.33457 / 0.34925 | 0.33484 / 0.34928 | 0.33473 / 0.34933 |
| BarrY | 0.16845 / 0.18887 | 0.16845 / 0.18888 | 0.16845 / 0.18881 |
| BarrZ | 0.08506 / 0.09617 | 0.08507 / 0.09618 | 0.08505 / 0.09613 |
| CRGrad | 0.48189 / 0.43857 | 0.48173 / 0.43847 | 0.48206 / 0.43831 |
| DeltaGamma | 0.03154 / 0.03182 | 0.03154 / 0.03197 | 0.03154 / 0.03196 |
| MuonNorm | — | 0.14916 / 0.15303 | 0.14914 / 0.15299 |
| DOMEff, IceAbs, IceScat, HoleIce | — | agree to the last digit | |

Wall time for all 16: ~93 s (Migrad default), ~34 s (Newton), each including ~10 s of
data loading. Migrad and Newton agree to < 0.1 % on every interval.

## 3.3 Tests

`ICTests`: 114 passed, 2 skipped (the Metal cases, which only run on macOS).
`AdaptiveGridTests` (scan algorithms): 44 passed. The linear-regression toy fit
reproduces the README's documented values bit for bit (non-gradient path unchanged).

---

# Part 4 — Findings

## 4.1 Are the single-fit errors now "correct"? Do you still need profile scans?

The Hesse errors are now an accurate measurement of the **curvature at the minimum**
— but that is a parabolic approximation. It equals the profile-likelihood interval
only when the likelihood is close to quadratic in that parameter.

| Parameter | Profile −/+ | Hesse |
|---|---|---|
| BarrZ | 0.0851 / 0.0962 | 0.090 |
| ConvNorm | 0.1027 / 0.1182 | 0.110 |
| DOMEff | 0.00257 / 0.00256 | 0.00257 |
| AstroNorm | **1.195 / 0.426** | 0.423 |
| SpectralIndex | **0.199 / 0.071** | 0.071 |
| PromptNorm | at bound / 2.60 | meaningless at the bound |

- **Hesse is enough** for nuisance parameters with a near-parabolic profile
  (detector parameters, ConvNorm, Barr, MuonNorm): it now matches the profile
  half-width to a few percent.
- **Hesse is not enough** for AstroNorm, SpectralIndex and PromptNorm. On the lower
  side AstroNorm trades off against PromptNorm once PromptNorm leaves its bound at 0,
  so the profile is strongly asymmetric. The curvature at the minimum only describes
  the narrow side. A parameter on a bound has no meaningful Hesse error at all.

Scans are now cheap (≈ 2–6 s per parameter after one free fit), so keep scanning the
parameters you report. Signs that a scan is needed: a parameter near a bound,
strongly correlated with one near a bound, or asymmetric crossings.

A further caveat applies to scans as well: Δχ² = 1 gives correct 68 % coverage only if
Wilks' theorem holds, which may not be the case near the PromptNorm bound (toys would
be needed to check).

## 4.2 Migrad's convergence criterion can be over-optimistic

At six fixed scan points, fits were run with tolerance 0.5 (EDM < 1e-3) and compared
with a tightly converged reference. "Gap" = how far above the true minimum the fit
stopped:

| Point | Migrad + gradient | Newton | Migrad, numerical (old) |
|---|---|---|---|
| MuonNorm | **0.109** (EDM 5e-4) | 0.0005 | **0.278** (EDM 5e-4) |
| ConvNorm | 0.0000 | 0.0004 | 0.0074 |
| BarrW | 0.0001 | 0.0004 | 0.0004 |
| AstroNorm | 0.0004 | 0.0005 | 0.0008 |
| SpectralIndex | 0.0001 | 0.0008 | 0.0004 |
| CRGrad | 0.0020 | 0.0004 | 0.0027 |

Migrad's EDM is computed from its *own* Hessian estimate, which it builds up by
updates and which can be wrong in some direction. It then reports EDM 5e-4 while
actually sitting 0.1–0.3 above the minimum — far more than the 0.02 that matters for
a crossing. This was true before this branch too (numerical path). Newton rebuilds its
model at every point, and its stops were always within 1e-3.

Consequence: with Minuit2, scan points must keep a tight tolerance — hence the
automatic `--scanTolerance` default (2.1). With the gradient this costs little (93 s
instead of 68 s for all 16 crossings). It is probably also why `--tolerance 1e-3` was
necessary in the first place.

## 4.3 SAY likelihoods do not recover the Asimov truth

Same Asimov sample, three likelihoods:

| | AstroNorm | MuonNorm | PromptNorm | SpectralIndex | -2lnL at minimum |
|---|---|---|---|---|---|
| Truth | 1.77 | 1.00 | 0.50 | 2.44 | |
| Poisson | 1.7700 (0.00σ) | 1.0000 | 0.5000 | 2.4400 | 2e-11 (exact) |
| SAYMean | 2.7236 (+2.25σ) | 0.6355 (−2.43σ) | 0.0000 | 2.3998 | 8244.25 |
| SAY (L_Eff) | 2.6759 (+2.18σ) | 0.6267 (−2.64σ) | 0.0000 | 2.4258 | 8244.03 |

Poisson recovers the truth exactly; both SAY variants are pulled by ~2σ in AstroNorm
and MuonNorm.

**Why.** "Asimov" sets each bin's count equal to its prediction, `k = mu`. For Poisson
this is exactly the maximum of the likelihood. For SAY it is not: `alpha` and `beta`
*both* depend on mu, and the derivative of the bin term at `k = mu` is not zero
wherever the MC variance `s` is large. The fit then moves the prediction to reduce it
in high-variance bins. (An earlier explanation in the discussion — that the SAYMean
mean is offset from mu — was wrong: for SAYMean the negative-binomial mean is exactly
mu; only L_Eff has an offset `s/mu`.)

**Relevance.** Asimov sensitivity studies with SAY are biased by this effect. Nothing
was changed; it is a physics question. One option would be "SAY-Asimov" data: choose
each bin's `k` such that the SAY term is stationary at the truth.

This is also why PromptNorm sits at its bound in these fits, and why the Gauss-Newton
Hessian was a poor error estimate here (2.7).

---

# Part 5 — How to use it

## Options

| Option | Default | Meaning |
|---|---|---|
| `--gradient` | `true` | Use the analytic gradient (IceCube). `false` = old numerical derivatives. |
| `--hessian` | `gn` | `gn`: Gauss-Newton Hessian as Migrad's metric (strategy raised to 2), reported errors from the exact Hessian. `none`: Minuit2's numerical second derivatives. |
| `--minimizer` | `Minuit2` | `Newton`: the new minimizer (needs `--gradient true --hessian gn`). |
| `--scanTolerance` | auto | Tolerance of scan points: `--tolerance` with Minuit2, 0.5 with Newton. |

Your existing command needs no changes:

```bash
LLHFit -c config_default.json --tolerance 1e-3 --fitRetries 10 --scanMode 1d-crossing --scanWorkers 1
```

To try Newton (≈ 2× faster again):

```bash
LLHFit -c config_default.json --tolerance 1e-3 --fitRetries 10 --scanMode 1d-crossing --scanWorkers 1 --minimizer Newton
```

To reproduce the old behaviour exactly: `--gradient false`.

## Validating on the MacBook (Metal)

The Metal kernels are written but could not be compiled here. On the Mac:

1. Build and run `ICTests`: the GPU tests run the Metal backend automatically
   (`AnalyticGradientTest.GpuMatchesCpu` compares the Metal derivative table with the
   CPU; FP32 tolerance 2e-4).
2. Run `GradientCheck -c <asimov config> --silent`. Away from the minimum, the
   `rel.diff` column should be at the level of the `fd.noise` column. At a minimum
   the gradient is ~0 and both columns are noise — use the start values, not a
   result file.

If Metal fails, the CPU path (no GPU) also has the gradient.

## Build notes

- Works with ROOT 6.28 through current master (6.41): version guards for the
  6.34 span signatures, `__has_include` for the 6.38 `FCNAdapter` merge.
- The `likelihood` library now links `ROOT::Minuit2` directly.

---

# Part 6 — What was not done

- **Slope-based crossing steps** (using the profile's slope, which the gradient
  provides for free, for Newton steps in the crossing search): not implemented —
  scan points became cheap enough that the remaining extra point per side matters
  little.
- **Newton as the default minimizer**: it is faster and its stops are more reliable,
  but it is new code. Migrad stays the default; this is your decision.
- **SAY-Asimov data** (4.3): a physics decision, not touched.
- **Metal**: untested (no compiler on Linux).

---

# Appendix A — Commits

| Commit | Content |
|---|---|
| `6774b06` | A1–A4: scan tolerance option, free-fit reuse, node 0 from free fit, bracket-width stop |
| `bd085c8` | B1/B2: analytic gradient, Gauss-Newton Hessian, exact final Hesse, GradientCheck |
| `7cc94b4` | CUDA zero-fill race fix |
| `367a027` | Newton minimizer; A5 step sizes |
| `9be9a9d` | README section on the new options |
| `6b8f37b` | Automatic scan tolerance per minimizer; ROOT < 6.34 build guards |

# Appendix B — Where things are in the code

| File | What it contains |
|---|---|
| `libraries/likelihood/IceCube/FluxGradient.{h,cpp}` | The 29 per-bin derivative sums: CPU loop, CUDA and Metal kernels |
| `libraries/likelihood/IceCube/SampleLikelihoodGradient.cpp` | Chain rule: derivative columns, bin terms, gradient and GN Hessian per sample |
| `libraries/likelihood/IceCube/BinTermDerivatives.h` | First/second derivatives of the SAY/Poisson bin terms in every branch |
| `libraries/likelihood/IceCube/Polygamma.h` | Digamma/trigamma and their cancellation-free differences |
| `libraries/likelihood/IceCube/ICLikelihood.{h,cpp}` | `calculate_gradient` / `calculate_hessian`, pulls, per-point cache |
| `libraries/likelihood/IceCube/GpuBinReduce.{h,cpp}` | `gather_all`: one-dispatch gather of many quantities |
| `libraries/likelihood/IceCube/CudaBackend.cpp` | The zero-fill race fix |
| `libraries/likelihood/Likelihood.h` | Gradient/Hessian interface (default: not available) |
| `libraries/likelihood/Fit.{h,cpp}` | GradFunctor, Hessian hookup, Minuit2 subclass, exact Hesse, step sizes, tolerance |
| `libraries/likelihood/NewtonMinimizer.{h,cpp}` | The Newton minimizer |
| `programs/LLHFit/ScanCommon.h/.cpp` | Free-fit reuse, node 0, scan tolerance, step sizes |
| `programs/LLHFit/CrossingScan1D.h` | `bracket_width` stop rule |
| `programs/GradientCheck/` | Validation tool |
| `programs/ictests/GradientTests.cpp`, `NewtonMinimizerTests.cpp` | Tests |

# Appendix C — Glossary

- **-2 ln L**: twice the negative log-likelihood; the number being minimised.
  Differences of 1 correspond to 1σ for one parameter.
- **Asimov data**: pseudo-data equal to the prediction at the true parameters.
- **Gradient**: vector of first derivatives of -2 ln L with respect to the parameters.
- **Analytic gradient**: computed from formulas, as opposed to finite differences.
- **Finite differences**: approximating a derivative by `(f(x+h) - f(x-h)) / 2h`.
- **Hessian**: matrix of second derivatives; its inverse (×2) is the covariance matrix.
- **Gauss-Newton Hessian**: approximation keeping only first derivatives of the
  prediction; exact where prediction = data.
- **Exact Hessian (here)**: finite differences of the analytic gradient.
- **EDM**: estimated distance to the minimum, `½ gᵀ H⁻¹ g`; convergence criterion.
- **Migrad**: Minuit2's quasi-Newton minimizer; builds its Hessian estimate iteratively.
- **Newton / Levenberg-Marquardt**: uses the supplied Hessian at every step; λ damps
  the step when the quadratic model is not trustworthy.
- **Active set**: the parameters currently held on their bounds.
- **Hesse**: Minuit's computation of the errors from the Hessian at the minimum.
- **Profile likelihood scan**: fix one parameter, minimise over the others, repeat.
- **Crossing search**: finds only the Δχ² = 1 points of the profile.
- **Lattice / node / unit**: scan points sit at `best fit + node × unit`,
  `unit = error/64`.
- **SAY, L_Eff, L_Mean**: likelihoods accounting for MC statistical uncertainty.
- **MC variance (s, ssq)**: `Σ w_i²` per bin, the estimate of that uncertainty.
- **Nuisance parameter**: a parameter not of interest (systematics), profiled over.
- **Pull / prior**: Gaussian penalty on a nuisance parameter.
- **FP64 / FP32**: double / single precision floating point.
- **Stream (CUDA)**: a queue of GPU operations executed in order.
