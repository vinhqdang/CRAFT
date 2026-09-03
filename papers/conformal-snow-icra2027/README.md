# Anytime-Valid Conformal Monitoring for Weather-Onset Detection

**Target venue:** ICRA 2027
**Status:** Prototype implementation in progress. The method (calibration, both betting rules, the spatial e-process, and the KITTI/nuScenes synthetic-corruption fallback) is implemented and unit-tested end-to-end in `conformal_monitor/`; it has not yet been run on the real evaluation datasets. Primary dataset (Snowy Scenes) access is still requested, not yet confirmed.

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
  - `corruption.py` — `WeatherOnsetStream`, the KITTI/nuScenes synthetic-corruption fallback pipeline (clear → onset → ramped severity), kept dataset-agnostic so real Snowy Scenes onset labels drop in later without redesign.
  - `evaluate.py` — wires the above to `craf_x.models.CRAFX_Net` into the detection-delay-vs-false-alarm operating-curve evaluation from plan.md.
- `tests/` — unit and integration tests for `conformal_monitor/` (`python papers/conformal-snow-icra2027/tests/run_all.py`).
- `scripts/run_operating_curve_experiment.py` — runs the full pipeline (calibration → onset stream → both bettors → operating curve) at a larger scale than the unit tests, as a mechanics smoke-run.

**Smoke-run caveat:** that script currently runs against an *untrained* `CRAFX_Net` (random init) and the mock dataset (fresh random tensors every call), and seeds neither, so its printed numbers vary from run to run — one run may show no alarms at all, another may show alarms (real or false) at some deltas. That variance is expected and not itself a bug: it confirms the pipeline executes correctly at scale (proper shapes, no crashes, well-formed output for every delta) but the specific numbers any single run prints are noise, not a finding about the method — don't quote them as if they were stable.

Not yet done: running any of this against a *trained* detector on real (or corruption-augmented real) data, tuning `kappa`/the calibration split, and the head-to-head operating-curve comparison against the Monroy Muñoz et al. baseline that the paper's central claim rests on.

## Primary dataset: Snowy Scenes

[Snowy Scenes](https://github.com/snowyscenes/dataset) (Ngo, Aksoy, et al., accepted at IEEE RA-L 2026) is a multimodal AV perception dataset collected in Espoo, Finland, purpose-built for snowy conditions: 22,331 synchronized frames over 14.4 km, 5,027 annotated LiDAR scans (128-beam LiDAR + RGB + 3 thermal cameras + GNSS/IMU), 221,081 3D bounding boxes across 27 semantic classes, with dedicated splits for accumulated snow, active snowfall, and highway snow.

This is a strong fit because it provides real, physically grounded adverse-weather distribution shift (clear → accumulating → active snowfall → highway snow) rather than requiring synthetic corruption of KITTI/nuScenes, and it already has published 3D detection/segmentation baselines (PointPillars, CenterPoint, TransFusion-L, Cylinder3D) to build the conformal wrapper on top of.

**Access status:** the dataset is not directly downloadable — access requires emailing the paper authors. A request has been sent; access is not yet confirmed. This is the single biggest schedule risk given the ICRA deadline.

**Fallback plan:** if access does not arrive in time, run the same pipeline against KITTI/nuScenes with synthetic weather corruption to simulate onset, and swap in real Snowy Scenes results if/when access comes through. The experimental section design is meant to be identical either way.

## Relationship to `craf_x`

This paper depends on `craf_x`: the CCP-informed betting rate is conditioned on the Cross-modal Consistency Probe (CCP) score from `craf_x/models/ccp.py`, reusing the sibling CRAF-X paper's disagreement signal as the covariate rather than reimplementing one from scratch, and `conformal_monitor/evaluate.py` runs the monitor directly on top of `craf_x.models.CRAFX_Net`. The 3D detector being monitored (currently CRAF-X; PointPillars/CenterPoint/TransFusion-L/Cylinder3D remain options) and the conformal-prediction + sequential-testing layer around it (`conformal_monitor/`) are kept paper-local. If that layer turns out to be broadly reusable beyond this paper, promote it into a shared location — see [`papers/README.md`](../README.md).

## Open risks to close before committing to the writeup

- **Dataset access** — pending author response; see fallback plan above.
- **Novelty claim** — narrowed and lit-checked as of September 2026 (see `plan.md`), but not exhaustively: re-run a fuller search (semantic search + citation chase) before the camera-ready contribution statement is locked in.
- **Competitive timing** — a co-author of the closest prior work (Monroy Muñoz, Verma & Timans, WACV 2026) has an existing line of work on conformal prediction for multi-object 3D detection in autonomous driving, and is well-positioned to make the AV/3D extension independently before ICRA 2027. Worth periodically re-checking their output.
