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

- `papers/craf-x-tvc/` — CRAF-X, the multimodal fusion detector (CCP/GAFM/ACT).
  **Background only in this paper.** Currently under review at *The Visual Computer*.
- `papers/conformal-snow-icra2027/` — the anytime-valid conformal monitoring layer,
  spatial e-BH monitoring, real Snowy Scenes evaluation, and the CCP-covariate negative
  finding. **This paper's own contribution**, extended with a second real dataset
  (CADC).

## Disclosure & Differentiation Note

This section exists because the user asked directly, and it is recorded here
deliberately rather than only in the cover letter, so the reasoning is auditable
independent of that document.

**The situation.** CRAF-X is under active review at a different journal, not yet
accepted or published. Building a new submission that describes the same architecture
risks looking like duplicate/concurrent submission of overlapping work, which most
journals (Elsevier included) prohibit, and which editors are trained to ask about
directly.

**A separate, more serious issue surfaced during drafting, and how it was handled.**
While reading the CRAF-X manuscript to write this paper's background section, its
Section 4 (Evaluation) was found to report specific mAP/NDS/attack-success-rate
numbers, a component-ablation table, a gating-threshold sensitivity sweep, and
cross-dataset (Waymo) generalization claims that do not trace to any executed code or
real data in this repository: `craf_x/datasets/nuscenes_dataset.py` and
`waymo_dataset.py` are confirmed dummy/mock loaders (no real nuScenes or Waymo data
exists on disk), no BEVFusion/TransFusion/CMT baseline implementation exists anywhere
in `craf_x/`, and the only script touching these numbers
(`papers/craf-x-tvc/scripts/generate_charts.py`) has them as literal hardcoded Python
constants used purely to draw bar charts, not computed from any experiment. This was
flagged directly to the user. The user's explicit, recorded decision: leave CRAF-X's
manuscript and its reported numbers completely untouched (this paper's authors did not
re-verify, correct, or otherwise act on this finding as part of this task), and write
this new paper's background section by describing CRAF-X's *architecture* (the
CCP/GAFM/ACT design, which is real, working code this repository actually runs and
trains — used directly, for instance, to produce this paper's own Snowy Scenes results)
without restating or citing CRAF-X's disputed quantitative performance claims as
established fact. That is exactly how Section 3.1 of the manuscript and the cover
letter are written: the architecture is described and cited to the companion
manuscript; no mAP/NDS/ASR number from that manuscript appears anywhere in this
submission.

**What this note is not.** This is not a determination that the overlap or the
disputed-numbers issue is fully resolved or safe to submit past. It is a record of what
was found, what was decided, and what was done about the decision that was reachable
within this task's scope (differentiating this paper's own content). The decision of
whether and how to address the CRAF-X manuscript itself — correct it, withdraw it, or
otherwise respond to its currently-active peer review — was explicitly declared out of
scope for this task and remains entirely the user's to make, on their own timeline,
outside of this paper.

## Open items before actual submission

- [ ] Confirm IMAVIS/VP-NAV's actual anonymity policy (author-block currently withheld
      as a placeholder; IMAVIS's Guide for Authors was not independently fetchable —
      confirm directly on the ScienceDirect page or via the submission portal before
      finalizing).
- [ ] Confirm IMAVIS's actual word/page-length guidance if any (not found in the
      publicly reachable search results used for this draft).
- [ ] Fill in Table 2 (CADC operating curve) once the parallel CADC experiment
      (tracked separately, not part of this paper's own file changes) completes.
- [ ] Fill in real author names/affiliations once the anonymity-policy question above
      is resolved.
- [ ] User's own decision, entirely separate from this paper: what to do about the
      CRAF-X manuscript's disputed Section 4 given it is currently under active review
      elsewhere.
