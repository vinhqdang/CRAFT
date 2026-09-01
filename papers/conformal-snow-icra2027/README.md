# Anytime-Valid Conformal Monitoring for Weather-Onset Detection

**Target venue:** ICRA 2027
**Status:** Planning — statistical model being designed; primary dataset access requested, not yet confirmed.

## Idea

Wrap a 3D object detector's outputs in split conformal prediction (calibrated on clear-weather data), then track an **e-process / testing-by-betting statistic** over the resulting per-frame miscoverage rate as a driving scene evolves. The statistic gives anytime-valid Type-I error control (via Ville's inequality) for detecting the *onset* of distribution shift — e.g. clear conditions degrading into snowfall — without needing a fixed monitoring horizon or multiple-testing correction. The target metric is detection delay (time from true weather onset to alarm) vs. false-alarm rate under stationary conditions, evaluated against naive fixed-window and corrected-batch conformal baselines.

See [`plan.md`](plan.md) for the full statistical setup, novelty considerations, and baselines as currently scoped.

## Primary dataset: Snowy Scenes

[Snowy Scenes](https://github.com/snowyscenes/dataset) (Ngo, Aksoy, et al., accepted at IEEE RA-L 2026) is a multimodal AV perception dataset collected in Espoo, Finland, purpose-built for snowy conditions: 22,331 synchronized frames over 14.4 km, 5,027 annotated LiDAR scans (128-beam LiDAR + RGB + 3 thermal cameras + GNSS/IMU), 221,081 3D bounding boxes across 27 semantic classes, with dedicated splits for accumulated snow, active snowfall, and highway snow.

This is a strong fit because it provides real, physically grounded adverse-weather distribution shift (clear → accumulating → active snowfall → highway snow) rather than requiring synthetic corruption of KITTI/nuScenes, and it already has published 3D detection/segmentation baselines (PointPillars, CenterPoint, TransFusion-L, Cylinder3D) to build the conformal wrapper on top of.

**Access status:** the dataset is not directly downloadable — access requires emailing the paper authors. A request has been sent; access is not yet confirmed. This is the single biggest schedule risk given the ICRA deadline.

**Fallback plan:** if access does not arrive in time, run the same pipeline against KITTI/nuScenes with synthetic weather corruption to simulate onset, and swap in real Snowy Scenes results if/when access comes through. The experimental section design is meant to be identical either way.

## Relationship to `craf_x`

This paper does not currently reuse the `craf_x` modeling code (CCP/GAFM/ACT are specific to the CRAF-X fusion architecture). It instead wraps arbitrary off-the-shelf 3D detectors (PointPillars, CenterPoint, TransFusion-L, Cylinder3D) in a conformal-prediction + sequential-testing layer. If that layer turns out to be broadly reusable across papers in this repository, it should move into a shared location (e.g. a new top-level module) rather than staying paper-local — see [`papers/README.md`](../README.md) for the "when to promote paper-local code to the shared library" guidance.

## Open risks to close before committing to the writeup

- **Dataset access** — pending author response.
- **Novelty claim** — online/adaptive conformal prediction and sequential e-process monitoring both have prior literature; it is plausible the exact combination has precedent. A literature search is needed before finalizing the headline contribution claim (most likely delta: block-dependence handling for temporally correlated driving frames, multi-object aggregation, and a real adverse-weather onset-detection evaluation — not the e-process-meets-conformal idea in the abstract, which is likely not novel by itself).
