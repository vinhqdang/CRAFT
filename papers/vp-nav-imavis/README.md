# VP-NAV (Image and Vision Computing): From Robust Fusion to Statistically Valid Monitoring

**Status:** First draft complete, not yet submitted. Targeting the *Image and Vision
Computing* (Elsevier) special issue "Visual Perception enabling Autonomous Navigation
(VP-NAV)", deadline 31 December 2026.

This paper synthesizes the repository's two other papers into one systems-level
contribution: a multimodal fusion detector engineered for cross-modal robustness
(background, from [`craf-x-tvc/`](../craf-x-tvc/), currently under review at *The
Visual Computer*), wrapped in a statistically principled anytime-valid conformal
monitor that detects real-world weather-onset degradation (this paper's own
contribution, extended from [`conformal-snow-icra2027/`](../conformal-snow-icra2027/)
with a second real adverse-weather dataset, CADC, evaluated concurrently with this
draft).

**Important — read [`plan.md`](plan.md) before submitting.** It records: (1) a real
academic-integrity finding surfaced while drafting this paper — the CRAF-X companion
manuscript's Evaluation section reports specific performance numbers that do not trace
to any executed code or real data in this repository — and the user's explicit decision
on how this paper handles that (background architecture only, no disputed numbers
restated); and (2) open items (anonymity policy, length guidance, the CADC results
table) that must be resolved before actual submission.

## Contents

- `manuscript/` — LaTeX source (`elsarticle` class) and compiled PDF.
- `plan.md` — venue research notes and the Disclosure & Differentiation Note.
- `cover_letter.txt` — draft cover letter disclosing the related under-review manuscript.

## Relationship to this repository's other papers

- Depends on [`craf-x-tvc/`](../craf-x-tvc/) for the monitored detector's architecture
  (background only — this paper does not restate or re-derive that paper's own
  experimental claims).
- Extends [`conformal-snow-icra2027/`](../conformal-snow-icra2027/) with a second real
  dataset (CADC) and journal-length depth; the ICRA paper's own submitted PDF is
  untouched by this work.
