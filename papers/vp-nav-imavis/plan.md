# Plan and Disclosure & Differentiation Note — VP-NAV (Image and Vision Computing)

## Target venue

*Image and Vision Computing* (Elsevier), Special Issue "Visual Perception enabling
Autonomous Navigation (VP-NAV)". Submission site: https://submit.elsevier.com/IMAVIS,
article type "VSI: VP-NAV". Deadline: 31 December 2026 (open for submission from
2026-09-01).

Format: `elsarticle` document class, `elsarticle-num` bibliography style (numbered
references) — confirmed available locally via the system LaTeX distribution. No hard
page-count cap was found in the publicly accessible parts of the Guide for Authors
(unlike ICRA's strict 8-page limit); IMAVIS's actual Guide for Authors page
(https://www.sciencedirect.com/journal/image-and-vision-computing/publish/guide-for-authors)
returned HTTP 403 to automated fetch, so its full author-facing text (word-count
guidance, required sections, anonymity policy for this specific special issue) was
**not independently confirmed** — see "Open items before actual submission" below.

## What this paper is

A synthesis of this repository's two existing papers into one systems-level
contribution:

- The monitored detector's architecture (a Cross-modal Consistency Probe, a Gated
  Adaptive Fusion Module, and an Adversarial Consistency Training objective) is
  described generically as background infrastructure. **This paper does not name or
  cite it as a specific external work.**
- `papers/conformal-snow-icra2027/` — the anytime-valid conformal monitoring layer,
  spatial e-BH monitoring, real Snowy Scenes evaluation, and the CCP-covariate negative
  finding. **This paper's own contribution**, extended with a second real dataset
  (CADC).

## Disclosure & Differentiation Note (historical — current manuscript carries none of
## this language; kept here as a record of the reasoning, not for submission)

**Current state (final decision):** the manuscript, cover letter, and bibliography name
and cite nothing about the detector beyond its architecture, described generically
("a camera/LiDAR fusion detector," "the monitored detector") — following the exact
precedent already established in the sibling `conformal-snow-icra2027/` (ICRA) paper,
which handles its own monitored detector the same way. No disclosure paragraph about
any other manuscript appears anywhere in this submission's materials, because nothing
in this submission names, cites, or depends on that other manuscript's specific
reported claims.

**How this note got here, for the record.** An earlier draft of this paper did name and
cite the detector's origin as a specific companion manuscript under review elsewhere,
with an accompanying disclosure paragraph in the cover letter. While drafting that
earlier version, a serious separate finding surfaced: that companion manuscript's own
Evaluation section reports specific mAP/NDS/attack-success-rate numbers, an
ablation table, and cross-dataset generalization claims that do not trace to any
executed code or real data anywhere in this repository (confirmed dummy/mock
nuScenes/Waymo dataset loaders, no baseline-model implementations, and the only script
touching these numbers hardcodes them as plot-generation constants, not experiment
output). That finding was flagged directly to the user at the time. The user's decision
then was to leave the other manuscript itself completely untouched, and to describe
only the real, working architecture as background without restating the disputed
numbers — which this paper did for one revision. The user's subsequent and current
decision, reflected in the manuscript as it now stands, is to remove the naming/citation
of that other manuscript entirely rather than disclose a relationship to it, so this
submission stands as fully independent. That decision is the user's own to make about
their own submission strategy; this note simply records that the disputed-numbers
finding was raised, and that whatever the user does about that other manuscript's
active peer review remains entirely outside the scope of this paper.

## Open items before actual submission

- [ ] Confirm IMAVIS/VP-NAV's actual anonymity policy (author-block currently withheld
      as a placeholder; IMAVIS's Guide for Authors was not independently fetchable —
      confirm directly on the ScienceDirect page or via the submission portal before
      finalizing).
- [ ] Confirm IMAVIS's actual word/page-length guidance if any (not found in the
      publicly reachable search results used for this draft).
- [x] Fill in Table 2 (CADC operating curve). Completed: 32 real CADC drives
      downloaded (39GB), detector trained from scratch (5 epochs, same protocol as
      Snowy Scenes), and the real operating-curve evaluation run twice -- a first,
      unstratified 2-drive calibration split gave a uniform false-alarm rate of
      1.00 (diagnosed as drive-to-drive heterogeneity correlated with collection
      date, not a size problem), fixed by stratifying the calibration split across
      dates (9 drives, 850 frames). With the fix, both bettors detect the real
      onset with zero false alarms; CCP-informed is tied or one frame faster than
      covariate-blind at every $\delta$ -- a modest, CADC-specific reversal of
      Snowy Scenes' negative result, reported as exactly that in Sections
      4.4-4.5 and Discussion 5.4, not as evidence the covariate now works in
      general. See `papers/conformal-snow-icra2027/plan.md`'s "Second real
      dataset: CADC" section for the underlying experiment log (dataset loader,
      training runs, calibration diagnostic).
- [ ] Fill in real author names/affiliations once the anonymity-policy question above
      is resolved.
- [ ] User's own decision, entirely separate from this paper: what to do about the
      CRAF-X manuscript's disputed Section 4 given it is currently under active review
      elsewhere.

## CRITICAL (2026-09-07): all reported numbers withdrawn — detector collapse

Every quantitative result previously recorded in this plan and in the
manuscript (Snowy Scenes operating curve, CADC operating curve, the three
CCP experiments, the mixture-betting evaluation) is **void** and must not be
cited. What follows is what was actually measured, not an interpretation.

**Finding.** Both detector checkpoints (`checkpoints/snowy_scenes_fixed/`,
`checkpoints/cadc/`) had collapsed to constant predictors:

- predicted heatmap std ~1e-3 inside a narrow band (~0.0404 Snowy,
  ~0.0142 CADC), and activation at ground-truth object cells is *not*
  higher than at empty cells (separation -6e-6 Snowy, -1.3e-4 CADC);
- no cell anywhere exceeds activation 0.05, so there were no detections at
  all — which is also why the phantom-detection score idea could not even
  be tested on these checkpoints;
- box-regression output |max| < 1.5e-3 everywhere.

**Decisive test.** Replacing the entire trained box head's output with
literal zeros leaves q_hat unchanged to 4 decimals and both regimes'
miscoverage identical to 5 decimals, on both datasets:

    Snowy  real 10.188355 / zeros 10.188589 ; nominal m 0.29940 both ; degraded m 0.14367 both
    CADC   real 10.755846 / zeros 10.756113 ; nominal m 0.07200 both ; degraded m 0.21105 both

So the nonconformity score ||B_pred - B_target||_1 had degenerated to
||B_target||_1: the monitor was measuring the ground-truth scene-content
distribution, not detector error. The two weather categories differ
substantially in object count (Snowy 2.2 -> 15.1 mean GT object cells;
CADC 5 -> 10), and that content shift — which co-varies with the weather
label — is what the reported "onset detections" were tracking. This also
explains why all three betting-rule experiments failed to beat the
baseline: there was no detector-degradation signal to bet on.

Reproduce with `papers/vp-nav-imavis/scripts/diagnose_detector_collapse.py`
(committed, unit-tested, real JSON outputs in `manuscript/`). Independently
reproduced by the coordinator before the fix was approved.

**Root cause.** `craf_x/utils/losses.py`'s `compute_det_loss` was a
placeholder — its own docstring said so ("Computes a mock detection loss",
`l_h = F.mse_loss(...)  # Should be Focal Loss`) — using MSE against a
heatmap target that is >99.99% zero, plus an *unmasked* L1 on box
regression. A constant is the optimum for both. Not an undertraining
problem: more epochs converge harder onto the constant.

**Fix (done).** `craf_x/utils/targets.py` (new): Gaussian-splatted heatmap
targets with CenterNet's `gaussian_radius` and a CenterPoint-style minimum
radius, plus dense box targets over the Gaussian core. `losses.py`:
penalty-reduced focal loss + object-masked L1 normalized by object count.
Both are strict generalizations at the object center (unit-tested).

**Acceptance gate before any new operating curve.** Re-run the
zeros-substitution test on each retrained checkpoint; it must show a
*material* difference in q_hat and per-object scores, and heatmap
activation must be clearly higher at ground-truth object cells than at
empty ones. No detection numbers get reported until that passes.

**Status.** Retraining in progress (separate checkpoint dirs —
`checkpoints/snowy_scenes_focal/`, `checkpoints/cadc_focal/` — the
collapsed checkpoints are kept for the before/after comparison, which is
itself worth reporting). The manuscript carries a withdrawal notice in the
abstract and at the head of Section 4. Methodological contributions (the
spatial e-process construction, the mixture-betting validity argument and
its unit tests) are unaffected; only the empirical numbers are void.

Scope note: `papers/conformal-snow-icra2027/` is out of scope per the
user's decision (already submitted, isolated) and has not been touched.

## Interpretation rule for the ablation, fixed BEFORE results are in

Recorded in advance deliberately: deciding how to read a result after
seeing it is how motivated reasoning gets in, and this project has already
been burned once by a measurement that looked like a finding and was
actually an artifact.

The retrained detectors are **undertrained** — 5 epochs on 3,016 frames
(Snowy), with training loss still descending at epoch 2 (64.9 -> 16.6 ->
13.0). Five epochs is a budget chosen for convenience, not a convergence
point. That confounds the two possible ablation outcomes **asymmetrically**,
so they do not get symmetric treatment:

- **Improvement beats the baseline** -> valid, and conservative.
  Undertraining works *against* the improvement, so an effect that survives
  it survived a handicap. Report it, with the undertraining caveat framed
  as a strength ("achieved despite an undertrained detector").
- **Improvement fails to beat the baseline** -> **confounded, not a
  finding.** "The method does not help" and "the detector is too weak for
  the method to have anything to work with" are indistinguishable from that
  measurement alone. Do NOT write this up as a negative result about the
  method. Flag it as confounded, report it as such, and settle whether a
  longer training run is warranted before drawing any conclusion.

This applies to all three variants (phantom-aware score, e-value merging,
lambda-mixture betting) on both datasets.

## Reproducibility note (for the paper's own reproducibility statement)

Training was done in chunks because a full run exceeds the ~1h
background-task ceiling in this environment: Snowy Scenes ran epochs 0-2,
was interrupted, and resumed from `checkpoint_epoch2.pth` for epochs 3-4.
`--resume` restores **model weights only** — Adam's first/second moment
estimates are not checkpointed, so each resumed chunk re-warms its
optimizer state over its first several steps. This is a small but real
deviation from an uninterrupted run and must be stated plainly in the
paper rather than glossed as "trained for 5 epochs".

## Per-model calibration in the quality-vs-signal curve (deliberate, state it)

The detector-quality-vs-monitorable-signal curve compares the onset jump in
m(t) across six checkpoints (collapsed, then focal epochs 0-4). **Each
checkpoint calibrates its own q_hat** on the same held-out calibration
frames, rather than all six sharing one common threshold.

This is a deliberate choice, not an oversight, and the writeup must name it
as such rather than leave a reviewer to find it and read it as a confound.
The justification: calibration is per-model in any real deployment -- you
calibrate the detector you are actually shipping -- so "each model under its
own calibration" is exactly the quantity a deployed monitor experiences. A
common fixed threshold across checkpoints would measure something no
deployment ever sees, and would additionally be dominated by the fact that
the collapsed checkpoint's score distribution lives on a completely
different scale (q_hat ~10.2 versus ~1.7 for the trained ones).

Consequence to state alongside it: because the quantiles differ, the
"content artifact subtraction" (jump_real - jump_zeros) is **indicative
only, not an identity**. There is no decomposition making that difference
the detector's own contribution, since the two branches threshold different
score distributions at different quantiles. The raw jump per checkpoint,
with its bootstrap CI, is the primary reported quantity; the subtraction is
reported for orientation and explicitly labelled as heuristic.

Goes in Method (or a footnote where the curve is reported).

## The null-detector control, and why it is a contribution rather than a check

Result (CADC, retrained detector, identical protocol, sole difference being
whether the box head's output is real or identically zero):

    real_detector  q_hat=1.935    d=0.30 FA=1.00 delay=-5.0   d=0.10 FA=1.00 delay=-2.0   d=0.05 FA=1.00 censored
    null_detector  q_hat=10.267   d=0.30 FA=0.00 delay= 6.0   d=0.10 FA=0.00 delay= 8.0   d=0.05 FA=0.00 delay=9.0

A detector that outputs nothing yields a perfectly behaved monitor: zero
false alarms, detection in 6/8/9 frames. The real detector's monitor is
broken. And 6/8/9 are **exactly** the numbers the manuscript reported as the
real CADC result (tab:cadc-results, covariate-blind). That is not
coincidence: the old CADC checkpoint was collapsed, so it *was* the null
detector. The control reproduces the published headline to the frame from a
model with no functioning box head.

### The inversion worth writing up

The null detector monitors CADC *better* than the real one, and the reason
is clean:

- With no detector, the score is pure scene content, which is relatively
  **homogeneous** across nominal drives -> calibration transfers -> FA=0.00.
- With a real detector, the score is genuine detector error, which is
  genuinely **heterogeneous** across sessions (measured: 5.3x spread in
  per-drive mean miscoverage, 0.118 to 0.623, all nominal clear-road drives
  on one collection date) -> calibration fails -> FA=1.00.

**The artifact was better-behaved than the truth.** This inverts the usual
intuition that cleaner numbers indicate a better method. The practical
warning, which is the most useful thing this paper can say: *a well-behaved
monitor is not evidence of a working one*, and the null-detector control is
what distinguishes them. The contribution is the diagnostic, not only the
fix -- it is cheap, needs no extra data or training, and any conformal
monitoring paper can run it.

Keep the null-detector row permanently in the results table for both
datasets, not as a one-off check.

### Attribution logic must be gated on false-alarm control

Recorded because it was a real bug, caught in the flattering direction:
the first version of `run_zeros_baseline.py` compared `mean_detection_delay`
numerically without checking false-alarm rate, and therefore reported a
monitor with FA=1.00 and a NEGATIVE delay (alarming before the onset
existed) as "real faster by 11.0 frames - attributable". Delay comparisons
between monitors are only meaningful when both control their false-alarm
budget (FA <= delta); otherwise the faster-looking arm may simply be firing
indiscriminately. Fixed, and the same gate applies to every variant
comparison in the ablation.

## Mondrian's worst-case advantage is checkpoint-dependent (do not overclaim)

On the 5-epoch CADC checkpoint, session-conditional calibration clearly
improved worst-case per-session coverage: worst |m - alpha| fell from
0.41-0.48 (marginal) to 0.19-0.20 (Mondrian) across all gaps, and that
improvement was the mechanism by which false-alarm control returned.

On the converged (20-epoch) checkpoint the ordering **reverses**:

    gap   marginal worst|m-a|   mondrian worst|m-a|
      0                0.1868                0.3444
     10                0.2211                0.3631
     20                0.2102                0.3530

Mondrian is now roughly 1.7x WORSE on worst-case per-session deviation,
while remaining at or slightly better than marginal on effect/noise. With
per-session quantiles fitted against a sharper detector, sessions appear to
diverge more rather than less. We do not have an explanation and are not
inventing one.

Consequences for the writeup:

- Lever 1's benefit must be stated as **observed on one checkpoint**, not
  as an unconditional property of session-conditional calibration. Any
  claim of the form "Mondrian improves worst-case per-session coverage"
  is contradicted by our own second measurement.
- The effect/noise comparison (0.55 -> 0.85 on the 5-epoch checkpoint) is
  unaffected by this and still holds as measured, but it too is a
  single-checkpoint result.
- This is worth reporting rather than burying: a calibration scheme whose
  benefit depends on the quality of the model being calibrated is a real
  and non-obvious caveat for anyone applying Mondrian conformal prediction
  to a deployed monitor.
