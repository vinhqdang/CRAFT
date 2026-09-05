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
- [ ] Fill in Table 2 (CADC operating curve) once the parallel CADC experiment
      (tracked separately, not part of this paper's own file changes) completes.
- [ ] Fill in real author names/affiliations once the anonymity-policy question above
      is resolved.
- [ ] User's own decision, entirely separate from this paper: what to do about the
      CRAF-X manuscript's disputed Section 4 given it is currently under active review
      elsewhere.
