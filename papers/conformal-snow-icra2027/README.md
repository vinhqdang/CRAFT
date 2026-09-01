# Anytime-Valid Conformal Monitoring for Weather-Onset Detection

**Target venue:** ICRA 2027
**Status:** Planning — statistical model being designed; primary dataset access requested, not yet confirmed.

## Idea

Wrap a 3D object detector's outputs in split conformal prediction (calibrated on clear-weather data), then track an **e-process / testing-by-betting statistic** over the resulting miscoverage rate as a driving scene evolves, to detect the *onset* of distribution shift (e.g. clear conditions degrading into snowfall) with anytime-valid Type-I error control (Ville's inequality) — no fixed monitoring horizon, no multiple-testing correction.

That backbone by itself is not novel — see the September 2026 literature check in [`plan.md`](plan.md). The closest prior work, [Monroy Muñoz, Verma & Timans (WACV 2026)](https://arxiv.org/abs/2602.12983), already does anytime-valid e-process failure detection for streaming vision, with the betting rate learned from the failure metric's own history. This paper's contribution is narrowed to what that work explicitly leaves open and what it doesn't address:

1. **Covariate-informed betting** — condition the betting rate on `craf_x`'s Cross-modal Consistency Probe (CCP) score (LiDAR/camera geometric disagreement) instead of only the failure metric's own history, to get a leading rather than lagging signal.
2. **Multi-object, spatially-resolved monitoring** — a grid of per-BEV-cell e-processes with anytime-valid multiplicity control, producing a live "where is perception untrustworthy" map, instead of one global/per-object scalar. The prior work is explicitly single-object and lists this as future work.
3. **Real adverse-weather distribution shift** in multimodal 3D driving perception, rather than 2D visual tracking.

The target metric is detection delay (time from true weather onset to alarm) vs. false-alarm rate, evaluated against the prior work's own covariate-blind betting rules (the real baseline, not a strawman) plus naive fixed-window and corrected-batch conformal baselines.

See [`plan.md`](plan.md) for the full statistical setup, the literature check with sources, and baselines as currently scoped.

## Primary dataset: Snowy Scenes

[Snowy Scenes](https://github.com/snowyscenes/dataset) (Ngo, Aksoy, et al., accepted at IEEE RA-L 2026) is a multimodal AV perception dataset collected in Espoo, Finland, purpose-built for snowy conditions: 22,331 synchronized frames over 14.4 km, 5,027 annotated LiDAR scans (128-beam LiDAR + RGB + 3 thermal cameras + GNSS/IMU), 221,081 3D bounding boxes across 27 semantic classes, with dedicated splits for accumulated snow, active snowfall, and highway snow.

This is a strong fit because it provides real, physically grounded adverse-weather distribution shift (clear → accumulating → active snowfall → highway snow) rather than requiring synthetic corruption of KITTI/nuScenes, and it already has published 3D detection/segmentation baselines (PointPillars, CenterPoint, TransFusion-L, Cylinder3D) to build the conformal wrapper on top of.

**Access status:** the dataset is not directly downloadable — access requires emailing the paper authors. A request has been sent; access is not yet confirmed. This is the single biggest schedule risk given the ICRA deadline.

**Fallback plan:** if access does not arrive in time, run the same pipeline against KITTI/nuScenes with synthetic weather corruption to simulate onset, and swap in real Snowy Scenes results if/when access comes through. The experimental section design is meant to be identical either way.

## Relationship to `craf_x`

This paper now depends on `craf_x` after all: the strengthened design (see above) conditions the e-process betting rate on the Cross-modal Consistency Probe (CCP) score from `craf_x/models/ccp.py`, reusing the sibling CRAF-X paper's disagreement signal as the covariate rather than reimplementing one from scratch. The 3D detector being monitored (PointPillars, CenterPoint, TransFusion-L, Cylinder3D, or CRAF-X itself) and the conformal-prediction + sequential-testing layer around it are paper-local. If that layer turns out to be broadly reusable beyond this paper, promote it into a shared location — see [`papers/README.md`](../README.md).

## Open risks to close before committing to the writeup

- **Dataset access** — pending author response; see fallback plan above.
- **Novelty claim** — narrowed and lit-checked as of September 2026 (see `plan.md`), but not exhaustively: re-run a fuller search (semantic search + citation chase) before the camera-ready contribution statement is locked in.
- **Competitive timing** — a co-author of the closest prior work (Monroy Muñoz, Verma & Timans, WACV 2026) has an existing line of work on conformal prediction for multi-object 3D detection in autonomous driving, and is well-positioned to make the AV/3D extension independently before ICRA 2027. Worth periodically re-checking their output.
