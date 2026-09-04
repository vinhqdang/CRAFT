# Anytime-Valid Conformal Monitoring for Weather-Onset Detection

**Target venue:** ICRA 2027
**Status:** Dataset access resolved; method implemented, unit-tested, and run end-to-end against the real archive on a Colab GPU (calibration, both bettors, the real onset stream). Not yet done: training the detector to convergence, so the numbers a real run currently produces are a mechanics check, not a result — see "Real Snowy Scenes evaluation" below.

## Idea

Wrap a 3D object detector's outputs in split conformal prediction (calibrated on clear-weather data), then track an **e-process / testing-by-betting statistic** over the resulting miscoverage rate as a driving scene evolves, to detect the *onset* of distribution shift (e.g. clear conditions degrading into snowfall) with anytime-valid Type-I error control (Ville's inequality) — no fixed monitoring horizon, no multiple-testing correction.

That backbone by itself is not novel — see the September 2026 literature check in [`plan.md`](plan.md). The closest prior work, [Monroy Muñoz, Verma & Timans (WACV 2026)](https://arxiv.org/abs/2602.12983), already does anytime-valid e-process failure detection for streaming vision, with the betting rate learned from the failure metric's own history. This paper's contribution is narrowed to what that work explicitly leaves open and what it doesn't address:

1. **Covariate-informed betting** — condition the betting rate on `craf_x`'s Cross-modal Consistency Probe (CCP) score (LiDAR/camera geometric disagreement) instead of only the failure metric's own history, to get a leading rather than lagging signal.
2. **Multi-object, spatially-resolved monitoring** — a grid of per-BEV-cell e-processes with anytime-valid multiplicity control, producing a live "where is perception untrustworthy" map, instead of one global/per-object scalar. The prior work is explicitly single-object and lists this as future work.
3. **Real adverse-weather distribution shift** in multimodal 3D driving perception, rather than 2D visual tracking.

The target metric is detection delay (time from true weather onset to alarm) vs. false-alarm rate, evaluated against the prior work's own covariate-blind betting rules (the real baseline, not a strawman) plus naive fixed-window and corrected-batch conformal baselines.

See [`plan.md`](plan.md) for the full statistical setup, the literature check with sources, and baselines as currently scoped.

## Contents

- `plan.md` — statistical design, literature check, and baselines.
- `conformal_monitor/` — the method's implementation:
  - `calibration.py` — split conformal calibration of craf_x detections (nonconformity scores, calibrated quantile, frame-level miscoverage rate `m(t)`).
  - `betting.py` — the e-process wealth process and the three betting rules: `AGRAPABettor` and `SFOGDBettor` (covariate-blind, reproducing the Monroy Muñoz et al. baseline) and `CCPInformedBettor` (this paper's primary contribution, plan.md claim 1).
  - `spatial.py` — the per-BEV-cell grid of e-processes with `"bonferroni"` and `"ebh"` (Wang & Ramdas e-BH) multiplicity-control modes, producing the live "where is perception untrustworthy" map (plan.md claim 2).
  - `corruption.py` — `WeatherOnsetStream`, the KITTI/nuScenes synthetic-corruption fallback pipeline (clear → onset → ramped severity). Superseded by `real_snow_stream.py` below for Snowy Scenes itself, but still useful for ablating "real vs. synthetic shift."
  - `real_snow_stream.py` — `RealSnowOnsetStream`, splicing real Snowy Scenes frames from a mild weather category (nominal) with frames from a severe category (degraded) at a known onset index. See "Real Snowy Scenes evaluation" below for why (no clear-weather split exists) and how categories were ordered by measured severity.
  - `evaluate.py` — wires the above to `craf_x.models.CRAFX_Net` into the detection-delay-vs-false-alarm operating-curve evaluation from plan.md. Device-aware (`calibrate_on_clear_weather`, `compute_global_wealth_trajectory`, `run_spatial_monitor` all move frames onto the model's device) and `calibrate_on_clear_weather` takes `num_workers` for datasets with slow `__getitem__` (e.g. zip-backed ones).
- `tests/` — unit and integration tests for `conformal_monitor/` (`python papers/conformal-snow-icra2027/tests/run_all.py`).
- `scripts/run_operating_curve_experiment.py` — runs the full pipeline (calibration → onset stream → both bettors → operating curve) at a larger scale than the unit tests, as a mechanics smoke-run.

**Smoke-run caveat:** that script currently runs against an *untrained* `CRAFX_Net` (random init) and the mock dataset (fresh random tensors every call), and seeds neither, so its printed numbers vary from run to run — one run may show no alarms at all, another may show alarms (real or false) at some deltas. That variance is expected and not itself a bug: it confirms the pipeline executes correctly at scale (proper shapes, no crashes, well-formed output for every delta) but the specific numbers any single run prints are noise, not a finding about the method — don't quote them as if they were stable.

Not yet done: running any of this against a *trained* detector on real (or corruption-augmented real) data, tuning `kappa`/the calibration split, and the head-to-head operating-curve comparison against the Monroy Muñoz et al. baseline that the paper's central claim rests on.

## Primary dataset: Snowy Scenes

[Snowy Scenes](https://github.com/snowyscenes/dataset) (Ngo, Raisuddin, et al., "Snowy Scenes: A Multimodal Multitask Dataset Toward Snow-Tonomous Vehicles") is a multimodal AV perception dataset purpose-built for snowy conditions.

**Access status: resolved.** The archive (`ROADVIEW5k.zip`, distributed via the authors' OneDrive) is downloaded and verified (5,027 frames total — 3,016 train / 1,005 val / 1,006 test, matching the paper's stated frame count exactly). It's ~49GB compressed / ~93GB uncompressed, so `CRAFXSnowyScenesDataset` (`craf_x/datasets/snowy_scenes_dataset.py`) reads directly out of the zip via `zipfile` rather than extracting it. Per-frame contents: RGB camera (`images/`), 128-beam LiDAR (`velodyne/`, same `(N,4)` float32 layout as KITTI), 3D object boxes already in the LiDAR frame (`object_labels/`, no camera↔LiDAR transform needed unlike KITTI), and per-point semantic segmentation (`labels/`, 29 SemanticKITTI-style classes from `label_map.yaml`) — the segmentation labels aren't used here since craf_x's head has no segmentation output.

### Real Snowy Scenes evaluation — a scoping correction from what was planned

Two things turned out different from the original plan once the real data was in hand:

1. **No clear-weather baseline exists.** The three splits (`accumulated`, `falling`, `highway`) are all already-snowy driving — there's no dry/clear split to calibrate on or to serve as the "before" side of an onset transition, and no per-frame severity label either.
2. **Within-category severity doesn't ramp monotonically.** Sampling the ground-truth per-point "snow" semantic-segmentation fraction across one `falling`-category sequence (ordered by real timestamp) shows real but noisy fluctuation (0.002–0.044), not a clean onset curve — active snowfall is bursty, not a steady accumulation.

What the data *does* give us: a real, physically grounded severity contrast **across** categories. Measuring the mean per-point "snow"-class fraction (ground truth) per category:

| category | mean snow-point fraction |
|---|---|
| `accumulated` | 0.0001 |
| `highway` | 0.0041 |
| `falling` | 0.0119 |

This ordering makes physical sense: `falling` captures active/airborne snowfall, which LiDAR registers as spurious near-range returns, while `accumulated` is settled snow on surfaces rather than airborne particles. `RealSnowOnsetStream` (`conformal_monitor/real_snow_stream.py`) uses this: it splices held-out `accumulated` frames (nominal/H0 regime) with `falling` frames (measured-most-severe regime) at a known onset index. Every frame on both sides of the transition is real sensor data — real images, real LiDAR, real labels — only the splice point itself is constructed, since Snowy Scenes' own sequences don't contain a within-sequence onset transition. This replaces the originally-planned "real Snowy Scenes" design (an in-dataset clear→snow transition) and the KITTI/nuScenes synthetic-corruption fallback (`corruption.py`, kept for ablation) with something better-grounded than the fallback and achievable with the data that actually exists.

**First live run** (`CRAFX_Net`, untrained, `bev_h=bev_w=64`, calibrated on 40 real `accumulated` frames, `alpha=0.2`): both the covariate-blind and CCP-informed bettors alarmed at `t=3`, before the real onset at `t=8` (`detection_delay=-5` for both). This is expected, not a finding: with random weights, box-regression residuals bear no relation to ground truth, so nearly every frame — calibration and test alike — looks like a miscoverage, and the monitor fires almost immediately regardless of true condition. It confirms the full pipeline (real dataset → real model on GPU → real calibration → real spliced stream → both bettors) runs correctly end to end; it says nothing about the method yet. That requires training the detector first.

Getting the real archive onto a GPU session and iterating against it surfaced two bugs no CPU-only, `num_workers=0` test had caught: `evaluate.py`'s three model-running functions never moved frame tensors onto the model's device (crashed immediately off CPU), and `CRAFXSnowyScenesDataset.__init__` cached an open zip handle that, under Linux's fork-based `DataLoader` workers, got shared (not re-opened) across worker processes and raced on its file offset (`BadZipFile`). Both are fixed with regression tests (device fix confirmed via a live GPU run, since this environment has no local CUDA to unit-test it against).

## Relationship to `craf_x`

This paper depends on `craf_x`: the CCP-informed betting rate is conditioned on the Cross-modal Consistency Probe (CCP) score from `craf_x/models/ccp.py`, reusing the sibling CRAF-X paper's disagreement signal as the covariate rather than reimplementing one from scratch, and `conformal_monitor/evaluate.py` runs the monitor directly on top of `craf_x.models.CRAFX_Net`. The 3D detector being monitored (currently CRAF-X; PointPillars/CenterPoint/TransFusion-L/Cylinder3D remain options) and the conformal-prediction + sequential-testing layer around it (`conformal_monitor/`) are kept paper-local. If that layer turns out to be broadly reusable beyond this paper, promote it into a shared location — see [`papers/README.md`](../README.md).

## Open risks to close before committing to the writeup

- **Detector training** — the paper's operating-curve claim needs a trained `CRAFX_Net`, not the current random-init smoke runs. This is now the critical path (dataset access and the real onset stream are done); see `tools/train.py --dataset snowy_scenes`.
- **Onset-transition framing** — the real spliced transition (`accumulated` → `falling`) is scientifically grounded (a measured, physically-explained severity contrast) but constructed, not a naturally-occurring in-dataset onset. Worth stating explicitly in the writeup rather than implying a found clear→snow sequence, and worth checking whether reviewers will accept a cross-category splice as "real distribution shift" — the alternative (splicing in real KITTI clear-weather frames as the nominal side instead of held-out `accumulated` frames) remains available if not, though this environment doesn't have real KITTI data downloaded to try it.
- **Novelty claim** — narrowed and lit-checked as of September 2026 (see `plan.md`), but not exhaustively: re-run a fuller search (semantic search + citation chase) before the camera-ready contribution statement is locked in.
- **Competitive timing** — a co-author of the closest prior work (Monroy Muñoz, Verma & Timans, WACV 2026) has an existing line of work on conformal prediction for multi-object 3D detection in autonomous driving, and is well-positioned to make the AV/3D extension independently before ICRA 2027. Worth periodically re-checking their output.
