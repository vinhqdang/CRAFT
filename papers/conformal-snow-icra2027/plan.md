# Model design notes — Anytime-Valid Conformal Monitoring for Weather-Onset Detection

Working notes from model-design discussion. A literature check was run in September 2026 (see below) to pressure-test the novelty claim before committing to a contribution statement — the earlier draft of this document treated "anytime-valid e-process over conformal miscoverage" as the contribution, which the check below shows is not defensible on its own.

## Literature check (September 2026)

Closest prior art found, from most to least direct:

1. **Monroy Muñoz, Verma & Timans, "Detecting Object Tracking Failure via Sequential Hypothesis Testing," WACV 2026 Workshops (Real-World Surveillance, 6th), March 2026** ([arXiv:2602.12983](https://arxiv.org/abs/2602.12983), HTML: [arxiv.org/html/2602.12983v1](https://arxiv.org/html/2602.12983v1)). This is the direct predecessor and the paper to differentiate against explicitly:
   - Frames tracking-failure detection as a sequential hypothesis test formalized as an e-process: `X_0 = 1`, `X_t = Π_{i=1}^{t} [1 + λ_i(ε − M_i)]`, where `M_t ∈ [0,1]` is a bounded tracking-quality metric and `ε` a tolerance threshold.
   - Betting rate `λ_t` is adaptive and *learned from the metric's own history* via aGRAPA (approximate GRAPA) or SF-OGD (Scale-Free Online Gradient Descent) — no external covariate is used.
   - Provides anytime-valid false-alert-rate control (Ville's inequality argument, same backbone we had planned).
   - Scope is explicitly **single-object, per-video** (tested on OTB-100, LaSOT, TrackingNet, GOT-10k; `N_vid=50`, one metric stream per video). No spatial structure, no multiplicity control across simultaneously-tracked objects.
   - Application is 2D visual tracking (surveillance/motion-capture/robotics); no LiDAR, no multimodal fusion, no 3D detection, no weather/distribution-shift treatment.
   - Its own stated future work: *"the broader methodology of sequential testing with e-processes is well suited to other streaming vision tasks, including multi-object tracking, action recognition, trajectory prediction, or anomaly detection"* — i.e., the multi-object extension is explicitly open, not claimed.
   - **Competitive-timing risk:** co-author Alexander Timans has an existing line of work on conformal prediction for multi-object 3D detection uncertainty in autonomous driving (bounding-box coverage guarantees). That group is well-positioned to make the AV/3D extension themselves before ICRA 2027. Worth re-checking their output every few months.

2. **State-dependent conformal prediction (Geng, Waite, Turnquist, Ivanov & Ruchkin, Dec 2025 / rev. Apr 2026)** ([arXiv:2512.02893](https://arxiv.org/abs/2512.02893)) — conditions conformal error bounds on the autonomous system's dynamical state, combined with symbolic reachability analysis. Covariate-conditioned, but **not sequential/anytime-valid** (batch conformal, no e-process/martingale), and no weather/distribution-shift treatment in the abstract.

3. **Context-aware nonconformity functions for robotic planning and perception (Kumar, Tayebati, Migliarba, Krishnan & Trivedi, Sept 2025)** ([arXiv:2509.21955](https://arxiv.org/pdf/2509.21955)) — learnable, context-conditioned nonconformity scores for robotics. Also **standard batch conformal, not sequential**; context comes from learned task representations, not cross-modal sensor disagreement specifically.

4. **Value-Gated Modality Refiner / "Before Fusion, Ask What to Keep" (Liu et al., June 2026)** ([arXiv:2606.02679](https://arxiv.org/html/2606.02679)) — uses cross-modal agreement/disagreement to gate fusion, but purely as a learned neural gate, not as a statistical test covariate, and targets sentiment/action-recognition/audio-visual tasks, not AV perception.

5. **Antonante, Spivak & Carlone, "Monitoring and Diagnosability of Perception Systems" (2020)** ([arXiv:2005.11816](https://arxiv.org/abs/2005.11816)) — foundational robotics perception-monitoring reference, but graph-theoretic/topos-theoretic diagnosability, not statistical sequential testing. Good related-work citation for "monitoring perception systems" framing at a robotics venue, not a competing method.

**Conclusion:** as of this check, nothing combines (a) sequential/anytime-valid e-process testing, (b) multi-object and spatially-resolved structure with multiplicity control, (c) a betting rule conditioned on an external cross-modal disagreement signal rather than the failure metric's own history, and (d) real adverse-weather distribution shift in multimodal 3D driving perception. Item 1 is the one to beat and cite explicitly; items 2–4 are adjacent-but-different and belong in related work. This is not a from-scratch literature review — re-run a fuller pass (semantic search + backward/forward citation chase from item 1) before the camera-ready novelty claim is locked in, and especially re-check item 1's group for new output given the competitive-timing risk noted above.

## Setup

Stream of driving scenes indexed by `t = 1, 2, …` (per LiDAR sweep or short frame-block). At each `t`, a base 3D detector produces predictions, wrapped in a conformal prediction region calibrated at target miscoverage level `α` using a held-out clear-weather calibration set (split conformal — standard, not the novel part).

For object `i` in frame `t`:
- Nonconformity score `s_i(t)` — e.g. `1 − IoU(predicted region_i(t), ground truth_i(t))`, or a conformalized box-regression residual for tighter geometric calibration.
- Coverage indicator `C_i(t) = 1{s_i(t) ≤ q̂_α}`, where `q̂_α` is the calibrated quantile from the clear-weather calibration set.
- Frame-level miscoverage rate: `m(t) = (1/n_t) Σ_i (1 − C_i(t))`.

## Null hypothesis

`H0`: the pipeline is operating nominally — `E[m(t) | F_{t-1}] ≤ α` for all `t`. Goal: detect the *first time* this is violated (snow-onset degradation), with **anytime-valid** Type-I control: `P_H0(∃t: reject at t) ≤ δ`, regardless of when or how often the process is checked. (This backbone — the e-process itself and its Ville's-inequality guarantee — is the established part; see literature check above. It is the engine, not the contribution.)

## The e-process (testing by betting)

Wealth process: `K_t = Π_{s=1}^{t} (1 + λ_s · (m(s) − α))`, with `λ_s ∈ [0, 1/(1−α)]` chosen `F_{s-1}`-measurably (predictable).

Under `H0`, `K_t` is a nonnegative supermartingale, so by Ville's inequality `P_H0(sup_t K_t ≥ 1/δ) ≤ δ`. Alarm the first time `K_t ≥ 1/δ`.

## Strengthened contribution — where the novelty actually needs to live

Given the literature check, the contribution is **not** "e-process applied to conformal miscoverage" (established, and item 1 above already does an anytime-valid sequential failure test for streaming vision with a learned betting rule). The delta needs to be concrete and checkable:

1. **Covariate-informed betting, not metric-history-only betting.** Item 1's aGRAPA/SF-OGD choose `λ_t` purely from the history of the failure metric `M_t` itself. Instead, condition `λ_t` on `craf_x`'s Cross-modal Consistency Probe (CCP) score — an external, physically-grounded signal (LiDAR/camera geometric disagreement) computed at the same timestep, not derived from the test statistic's own past. Claim: this sharpens detection power (shorter onset-to-alarm delay at the same false-alarm budget `δ`) versus a covariate-blind bettor (aGRAPA/SF-OGD applied to `m(t)` alone), because the bettor can act on a leading indicator of degradation rather than only the lagging miscoverage signal itself. This is the paper's primary empirical claim and must be validated head-to-head against item 1's own betting rules as the baseline, not a strawman.
2. **Multi-object aggregation with spatial resolution, not a single per-video/per-frame scalar.** Item 1 is explicitly single-object/per-video and lists multi-object extension as open future work. Run the e-process per BEV grid cell (or cell-cluster) rather than one global statistic per frame, with anytime-valid multiplicity control across cells (online e-BH / closed e-testing — see e.g. the online-FDR-with-e-values line of work) so simultaneously testing many spatial regions doesn't inflate the false-alarm rate. Output is a live, spatially localized "where is perception currently untrustworthy" map, not one global alarm — this is the actionable, robotics-relevant deliverable (e.g., feeding a downstream controller or `craf_x`'s GAFM gate) and is what makes this an ICRA paper rather than a stats paper.
3. **Temporal dependence within a scene.** The martingale argument doesn't require independence, only `E[m(t) | F_{t-1}] ≤ α` under `H0`, which conformal calibration gives regardless of within-scene correlation. Worth stating explicitly and showing empirically that block-level aggregation (short scene chunks, ~10–20 frames) trades detection latency against noise reduction — this affects power, not validity.
4. **Weather-onset detection as the target quantity, on real distribution shift.** Evaluate on detection delay (frames/seconds from snow onset to alarm) vs. false-alarm rate under stationary clear-weather, as an operating curve, on real snow-onset transitions (Snowy Scenes) rather than only synthetic corruption — this is the ICRA-relevant metric and the reason a real adverse-weather dataset matters over KITTI/nuScenes alone.

## Baselines to beat

- **Monroy Muñoz et al. (WACV 2026) e-process, as-is**, applied to the frame-level miscoverage rate `m(t)` with aGRAPA or SF-OGD betting (covariate-blind). This is the real baseline for claim 1 above, not a strawman.
- Fixed-window batch conformal test recomputed every `K` frames, uncorrected (should show inflated false alarms under continuous monitoring — the naive-practitioner failure mode).
- Same, with Bonferroni / O'Brien-Fleming correction (valid but conservative — should show higher detection latency).
- Raw AP/mIoU with a CUSUM change-detector (no distribution-free guarantee — informal baseline).
- Single global e-process (no spatial resolution) vs. the per-cell version, to isolate the value of claim 2 above.

## Parameters to fix

- `α` (target miscoverage, e.g. 0.1)
- `δ` (false-alarm budget, e.g. 0.05)
- Calibration split: no clear/no-snowfall scenes exist in Snowy Scenes (see below) — currently held-out `accumulated`-category frames (measured mildest), not a true clear baseline.
- Dataset access resolved; ordering metadata resolved too — see "Snowy Scenes scoping notes" below.
- BEV cell size / cell-cluster granularity for the spatial e-process grid, and the online-FDR level for cross-cell multiplicity control

## Snowy Scenes scoping notes (resolved access, revised design)

Access arrived (`ROADVIEW5k.zip`, ~49GB, read in-place via `zipfile` — see `papers/conformal-snow-icra2027/README.md` for the full access story). Two assumptions in the original dataset decision below turned out wrong once the real data was inspected:

1. **No clear-weather baseline.** All three splits (`accumulated`, `falling`, `highway`) are already-snowy driving; there's no dry/clear split to calibrate on.
2. **No within-sequence onset ramp.** Sampling the ground-truth per-point "snow" class fraction across a timestamp-ordered `falling` sequence shows noisy fluctuation (bursty active snowfall), not a clean monotonic onset curve.

What's real and usable instead: the mean per-point "snow" fraction differs sharply and physically-sensibly **across** categories (`accumulated` 0.0001 < `highway` 0.0041 < `falling` 0.0119 — active/airborne snowfall registers as spurious near-range LiDAR returns, settled snow doesn't). `conformal_monitor/real_snow_stream.py`'s `RealSnowOnsetStream` splices held-out `accumulated` frames (nominal) with `falling` frames (degraded) at a known onset index — real sensor data on both sides, only the splice point is constructed. This replaces the KITTI/nuScenes synthetic-corruption fallback (`corruption.py`, kept for a real-vs-synthetic ablation) as the primary evaluation stream. Flagged as an open risk in the README: reviewers may push back on a cross-category splice standing in for a natural onset transition — a real-KITTI-clear-frames splice remains a fallback option if so, not yet tried since this environment has no real KITTI data downloaded.

## Dataset decision (superseded — kept for history)

Primary target is Snowy Scenes (real adverse-weather distribution shift, published 3D detection baselines to build on). Access is requested but not confirmed as of this writing, and the ICRA deadline does not allow waiting indefinitely. Do not block model/method development or writing on it:

- Keep the method and evaluation protocol dataset-agnostic (calibration split + onset-labeled evaluation stream) so either dataset drops in without redesign.
- Build the KITTI/nuScenes-with-synthetic-corruption pipeline in parallel now, not as an afterthought, so there's a working end-to-end result regardless of how the access request resolves.
- Swap in real Snowy Scenes results if/when access arrives; keep the fallback results too, since real vs. synthetic shift is itself a useful robustness comparison to report either way.

## Next steps

- [ ] Fuller literature pass (semantic search + citation chase from Monroy Muñoz et al. 2602.12983) before locking the camera-ready novelty claim; re-check that group's output periodically given the competitive-timing risk.
- [x] Follow up on Snowy Scenes dataset access request — resolved; see "Snowy Scenes scoping notes" above.
- [x] Stand up the KITTI/nuScenes synthetic-corruption fallback pipeline — `conformal_monitor/corruption.py` (`WeatherOnsetStream`), unit-tested in `tests/test_corruption.py`.
- [x] Implement the covariate-blind baseline (Monroy Muñoz et al.'s aGRAPA/SF-OGD on `m(t)`) first, to have a real comparison point — `conformal_monitor/betting.py` (`AGRAPABettor`, `SFOGDBettor`).
- [x] Implement the CCP-informed betting variant and the per-BEV-cell spatial e-process with online multiplicity control — `conformal_monitor/betting.py` (`CCPInformedBettor`) and `conformal_monitor/spatial.py` (`SpatialEProcessGrid`, Bonferroni and e-BH modes).
- [x] Design the detection-delay-vs-false-alarm operating-curve evaluation — `conformal_monitor/evaluate.py` (`operating_curve`), wired to `CRAFX_Net` and unit-tested on the synthetic-corruption stream.
- [x] Implement the real Snowy Scenes onset stream and run the pipeline against the real archive on a Colab GPU (calibration + both bettors) — `conformal_monitor/real_snow_stream.py`. Surfaced and fixed two real bugs (`evaluate.py` device handling, a fork-unsafe cached zip handle in `CRAFXSnowyScenesDataset`) that the CPU-only unit test suite couldn't have caught.
- [x] **Train `CRAFX_Net` on Snowy Scenes** (`tools/train.py --dataset snowy_scenes`) — done, 5 epochs on the real `train` split (local GPU), loss converged (~2.3 → ~0.008). Checkpoints in `checkpoints/snowy_scenes/`.
- [x] **Fix the CCP-collapse regression this uncovered** — fixed in `craf_x/training/adversarial.py` (commit `ed9ed91`): the two adversarial branches' CCP scores are now also supervised by `compute_ccp_loss` with a mismatched (`m=0`) target, alongside the existing clean-branch call. Retrained; the fix made `S` vary again but in the wrong direction for the task (higher disagreement in the nominal regime than the degraded one) — a second real negative finding, not a fix that restored the hoped-for advantage. Both findings are written up in the submitted manuscript (see below) as an honest, mechanistically-diagnosed open problem, not a validated claim.
- [x] **Manuscript submitted to ICRA 2027** — locked for double-blind review as of this writing. `papers/conformal-snow-icra2027/manuscript/` is frozen; do not edit it further until reviews come back. The paper's headline contributions are the spatial multiplicity-control construction and the real Snowy Scenes evaluation protocol/results; covariate-informed betting is reported as an honest, fully-diagnosed negative finding rather than a third claim.
- [ ] Tune `kappa` (the CCP-disagreement gain in `CCPInformedBettor`) and the calibration split size; both are currently unfit placeholders. Blocked on the CCP-collapse fix above — tuning `kappa` against a covariate that never varies is meaningless.

## Second real dataset: CADC (Canadian Adverse Driving Conditions)

Post-submission R&D for the next iteration of this work (a future revision or extended version) — validating the method on more than one real adverse-weather dataset. `papers/conformal-snow-icra2027/manuscript/` is submitted/locked and untouched by any of this.

**Why CADC**: multimodal (camera + LiDAR, unlike the LiDAR-only WADS), already cited in the submitted manuscript's Related Work (`Pitropov2021CADC`) as the closest adjacent snow dataset, and the published Snowy Scenes paper itself frames CADC and WADS as the two closest prior snowy-conditions datasets — extending it is a natural next step, not a new direction from nowhere.

**Access and format, researched (not guessed) before downloading anything**: official site `cadcd.uwaterloo.ca`, devkit at `github.com/mpitropov/cadc_devkit`. Verified the devkit's actual `download_cadcd.py` source directly (not a summary) — base URL `http://wiselab.uwaterloo.ca/cadcd_data/`, per-date `calib.zip`, per-drive `labeled.zip` + `3d_ann.json`, 70 labeled drives across three collection dates (`2018_03_06`, `2018_03_07`, `2019_02_27`). Confirmed the server is live and supports resumable range requests via a real `HEAD` request before committing to anything. Read the actual CADC paper (IJRR 2021 / arXiv:2001.10117) for the annotation schema (JSON array of frames, each a list of cuboids with `position`/`dimensions`/`yaw` already in the LiDAR frame — no camera↔LiDAR transform needed, same as Snowy Scenes) and confirmed LiDAR points are `(N,4)` float32 `[x,y,z,intensity]`, identical to KITTI/Snowy Scenes, by reading the devkit's own `lidar_utils.py`.

**A real finding worth having checked**: the devkit's own `cadc_dataset_route_stats.csv` exposes genuine per-drive metadata — a `Road snow cover` column (`None` for every 2018_03_06/2018_03_07 drive, `Covered` for every 2019_02_27 drive) and a quantitative `Snow points removed` severity count (matching the paper's light/medium/heavy/extreme snowfall bins). This is a **real, dataset-provided nominal-vs-degraded contrast** — unlike Snowy Scenes, which shipped no such per-drive label at all and required deriving severity ourselves from semantic-segmentation class fractions across categories that turned out not to correspond to a real onset transition. If CADC's `bare`-vs-`covered` split holds up as a real detectable contrast in our evaluation, it is a stronger evaluation design than Snowy Scenes' constructed accumulated→falling splice, not just a second data point.

**Scoping decision, disclosed rather than hidden**: full CADC labeled data is ~93GB across 70 drives. Downloaded a representative ~40.5GB subset instead — all 18 "bare road" drives (2018_03_06 + 2018_03_07, ~22.8GB) plus 14 "snow-covered" drives spread across 2019_02_27's 57 available drives (~17.8GB) — comparable in scope to the Snowy Scenes download, for time/disk budget reasons, not because the rest of the dataset is unavailable.

Progress (updated as this proceeds):
- [x] Research CADC's real access method, format, and per-drive metadata (above).
- [x] Write `scripts/download_cadc.sh` (same outer-retry-loop lesson as `download_snowy_scenes.sh`: a fresh curl process per attempt, not curl's own `--retry`, which we proved silently discards resumed progress on this kind of connection).
- [x] Write `craf_x/datasets/cadc_dataset.py` (`CRAFXCADCDataset`) and `tests/test_cadc_dataset.py` (10 tests, synthetic fixtures). Verified against real partially-downloaded data (drive `2019_02_27/0002`, 71 real frames) before finalizing, which is how a real discrepancy from the paper's own Fig. 7 diagram was caught: the actual extracted `labeled.zip` content sits under an extra `labeled/` subfolder the diagram doesn't show. `sample_indices` are formatted `f"{category}_{date}_{drive}_{frame:010d}"` (`category` ∈ `{bare, covered}`) specifically so `conformal_monitor.real_snow_stream`'s `category_indices`/`category_subset` (written for Snowy Scenes) work against this dataset completely unchanged — confirmed with an integration test, not just asserted.
- [x] Wire `CRAFXCADCDataset` into `tools/train.py` (`--dataset cadc`); smoke-tested for 3 real optimizer steps against the already-downloaded `2019_02_27` drives (238 real frames from 3 drives) on GPU — finite loss, checkpoint saved correctly.
- [ ] Full ~40.5GB download (in progress in the background as of this writing; ~3.7GB downloaded, 3 of 14 planned `covered` drives fully extracted, 0 of 18 planned `bare` drives yet).
- [ ] Train `CRAFX_Net` on CADC from scratch (own config/num_classes, smoke-tested — see above).
- [x] Write `scripts/run_real_operating_curve_cadc.py` (CADC counterpart of `run_real_operating_curve.py`): uses the real `bare`-vs-`covered` road-condition split rather than a constructed splice, and — since CADC has multiple drives per category, unlike Snowy Scenes' pooled frames — holds out calibration frames from *disjoint whole bare drives* rather than a frame-count split within one pooled category, avoiding a temporal-adjacency leakage concern the Snowy Scenes driver had accepted. The drive-parsing helper (`_drives_for_category`) was verified against the real partially-downloaded data (238 real `covered` frames split correctly across its 3 real drives) before being relied on for anything.
- [ ] Run it once training is done and enough `bare` drives have downloaded; report the real numbers honestly, whatever they are — including if this reproduces the CCP-informed-betting negative finding, or if it doesn't.
