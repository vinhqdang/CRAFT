# Model design notes — Anytime-Valid Conformal Monitoring for Weather-Onset Detection

Working notes from initial model-design discussion. Not yet reviewed against literature — see "Open risks" in `README.md`.

## Setup

Stream of driving scenes indexed by `t = 1, 2, …` (per LiDAR sweep or short frame-block). At each `t`, a base 3D detector produces predictions, wrapped in a conformal prediction region calibrated at target miscoverage level `α` using a held-out clear-weather calibration set (split conformal — standard, not the novel part).

For object `i` in frame `t`:
- Nonconformity score `s_i(t)` — e.g. `1 − IoU(predicted region_i(t), ground truth_i(t))`, or a conformalized box-regression residual for tighter geometric calibration.
- Coverage indicator `C_i(t) = 1{s_i(t) ≤ q̂_α}`, where `q̂_α` is the calibrated quantile from the clear-weather calibration set.
- Frame-level miscoverage rate: `m(t) = (1/n_t) Σ_i (1 − C_i(t))`.

## Null hypothesis

`H0`: the pipeline is operating nominally — `E[m(t) | F_{t-1}] ≤ α` for all `t`. Goal: detect the *first time* this is violated (snow-onset degradation), with **anytime-valid** Type-I control: `P_H0(∃t: reject at t) ≤ δ`, regardless of when or how often the process is checked.

## The e-process (testing by betting)

Wealth process: `K_t = Π_{s=1}^{t} (1 + λ_s · (m(s) − α))`, with `λ_s ∈ [0, 1/(1−α)]` chosen `F_{s-1}`-measurably (predictable).

Under `H0`, `K_t` is a nonnegative supermartingale, so by Ville's inequality `P_H0(sup_t K_t ≥ 1/δ) ≤ δ`. Alarm the first time `K_t ≥ 1/δ`. This gives valid Type-I error at any stopping time — no Bonferroni correction, no fixed monitoring horizon needed.

For `λ_s`: start with the Waudby-Smith–Ramdas running-mean-estimate betting rule rather than a fancier online-learning scheme (ONS, GRAPA) — lower implementation risk given the deadline. Revisit only if there's time.

## Where the actual contribution needs to live

The e-process-meets-conformal idea by itself is likely not novel (see "Open risks" in the README). The contribution needs to come from handling the driving-scene structure:

1. **Temporal dependence.** Frames within a scene are highly correlated — the martingale argument doesn't require independence, only `E[m(t) | F_{t-1}] ≤ α` under `H0`, which conformal calibration gives regardless of within-scene correlation. This is a real strength worth stating explicitly, but it affects *power*, not *validity* — should show empirically that block-level aggregation (short scene chunks, ~10–20 frames) trades detection latency against noise reduction.
2. **Multi-object aggregation.** Frames with more detected objects shouldn't dominate the statistic — weight per-object miscoverage rather than a flat per-frame average.
3. **Weather-onset detection as the target quantity.** Evaluate on detection delay (frames/seconds from snow onset to alarm) vs. false-alarm rate under stationary clear-weather, as an operating curve — this is the ICRA-relevant metric, not just coverage accuracy.

## Baselines to beat

- Fixed-window batch conformal test recomputed every `K` frames, uncorrected (should show inflated false alarms under continuous monitoring — the naive-practitioner failure mode).
- Same, with Bonferroni / O'Brien-Fleming correction (valid but conservative — should show higher detection latency).
- Raw AP/mIoU with a CUSUM change-detector (no distribution-free guarantee — informal baseline).

## Parameters to fix

- `α` (target miscoverage, e.g. 0.1)
- `δ` (false-alarm budget, e.g. 0.05)
- Calibration split from clear/no-snowfall scenes (Snowy Scenes, if access is granted in time)
- Pending dataset access and its ordering metadata: whether scenes carry real temporal ordering across the full drive, or need to be concatenated synthetically to simulate weather onset

## Dataset decision (open, deadline-sensitive)

Primary target is Snowy Scenes (real adverse-weather distribution shift, published 3D detection baselines to build on). Access is requested but not confirmed as of this writing, and the ICRA deadline does not allow waiting indefinitely. Do not block model/method development or writing on it:

- Keep the method and evaluation protocol dataset-agnostic (calibration split + onset-labeled evaluation stream) so either dataset drops in without redesign.
- Build the KITTI/nuScenes-with-synthetic-corruption pipeline in parallel now, not as an afterthought, so there's a working end-to-end result regardless of how the access request resolves.
- Swap in real Snowy Scenes results if/when access arrives; keep the fallback results too, since real vs. synthetic shift is itself a useful robustness comparison to report either way.

## Next steps

- [ ] Literature search to confirm the actual novelty delta before finalizing the contribution statement.
- [ ] Follow up on Snowy Scenes dataset access request.
- [ ] Stand up the KITTI/nuScenes synthetic-corruption fallback pipeline.
- [ ] Implement the WSR betting-rule e-process and the three baselines above on whichever dataset is ready first.
