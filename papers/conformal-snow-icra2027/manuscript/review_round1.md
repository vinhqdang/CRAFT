# Review Round 1 (pre-results-completion pass)

Scope: main.tex + subfiles, bibliography. Results section (Sec. 5/sec:results)
intentionally placeholder pending retrain — not evaluated for factual accuracy
here.

## Journal-Fit Reviewer

**Fit for ICRA 2027:** Good topical fit — anytime-valid monitoring for AV
perception under real weather shift is squarely in scope, and the paper is
honest rather than promotional about validation gaps, which robotics
reviewers tend to reward over confident overclaiming.

**Critical:** `2relate.tex` describes the monitored detector as "a companion
piece of this repository's own prior work" and the bibliography entry
(`CRAFXTechReport2026`) literally embeds a local repo path
(`papers/craf-x-tvc/`). If ICRA 2027 runs double-blind review (current IEEE
ICRA policy — verify against the actual call for papers before submission),
this deanonymizes the authors immediately. Must fix regardless: reads
unprofessionally even under single-blind review.

**Major:** The abstract states the spatial monitoring "turn[s] one global
alarm into a live, spatially localized map" without flagging that this
specific claim is validated only via unit tests and mock data, not against a
real multi-object driving scene (Sec. 5.4/conclusion is honest about this,
but the abstract oversells relative to the body — abstracts get skimmed and
should not need the fine print to avoid overclaiming).

## Peer Reviewer 1 (Methodology)

**Major:** n=5 onset replicates and n=5 stationary-control replicates is a
very small sample for an operating-curve claim. No confidence intervals, no
variance, no significance test comparing the two bettors' detection-delay
distributions is reported or planned. Even if the corrected checkpoint shows
a numeric gap between bettors, at n=5 that gap could easily be noise. The
paper should either (a) increase replicate count substantially before
submission, or (b) report bootstrap CIs / a paired test (the streams are
already evaluated pairwise per-delta per `evaluate.py`'s own docstring —
this pairing should be stated explicitly and exploited statistically, not
just architecturally) and be explicit that the current n is a pilot-scale
demonstration, not a powered comparison.

**Major:** Calibration set size (n=40) is small for a finite-sample
conformal quantile at $\alpha=0.2$ — worth a sentence acknowledging the
coverage guarantee's finite-sample looseness at this n, rather than treating
$\hat q_\alpha$ as if it were asymptotically exact.

**Minor:** $\kappa=2.0$ and the calibration split size are stated as
untuned defaults (good, honest), but the paper never states what a
*sensitivity analysis* over $\kappa$ would look like even qualitatively —
worth one sentence on what varying $\kappa$ is expected to do to the
false-alarm/delay tradeoff, even without running it.

## Peer Reviewer 2 (Domain / Related Work)

**Minor:** Related work differentiates against Monroy Muñoz et al. clearly
and substantively (not a strawman) — good. Geng et al., Kumar et al., Liu et
al., Antonante et al. are each given a specific, correct reason they don't
overlap. No missing directly-competing 2025-2026 work identified in this
pass.

**Minor:** `SnowyScenes2026` bib entry visibly notes that full bibliographic
metadata (author list, volume/issue/pages) could not be verified due to
IEEE Xplore blocking automated fetches. This is honest but should be
resolved (manually, by the human author, against the actual publisher
record) before submission, not left as a visible disclaimer in a submitted
PDF — reads as unpolished even though the underlying caution is correct
practice.

## Peer Reviewer 3 (Perspective / Practical Impact)

**Minor:** The CCP-collapse negative-finding narrative (Sec. 4.3) is a
genuine strength — it's rare and valuable for a paper to report a
methodological failure this candidly, and it likely has value to the
broader community reusing internal robustness signals as external
covariates, independent of this paper's specific results. Recommend keeping
this section's honesty exactly as-is through revision; don't let later
polish passes soften it into vagueness.

**Minor:** No discussion of what a downstream planner would actually do with
the spatial "untrustworthy region" map once the paper delivers one —
worth one sentence of practical grounding (e.g., feeding GAFM's own gating,
already gestured at in the plan.md source material) even at the level of
motivation rather than a full integration.

## Devil's Advocate

**CRITICAL:** The paper's entire framing is built around three claims, the
first and most emphasized of which (covariate-informed betting beats
covariate-blind) is *currently unknown* pending the retrain. If the
corrected checkpoint still shows no advantage, the current Introduction and
Abstract oversell what the paper will actually be able to conclude — they
are written as if the mechanism is expected to work, with the collapse
story framed as a solved detour rather than a possible dead end. The
manuscript MUST be prepared to genuinely reframe (not just append a caveat)
if the corrected results still show no advantage: honest failure-mode
papers are publishable and valuable, but only if the framing throughout
(not just one results paragraph) reflects that reality. This is not a
"different data path may show a decision-bearing regression" hypothetical —
it is unresolved as of this review and gates everything downstream.
**Adjudication: validated, blocks Accept until Results/Discussion/Conclusion
are actually written against the real corrected numbers, whichever way they
land — tracked, not yet resolvable this round.**

**Major:** No figure in the entire manuscript. For a robotics venue with a
strong visual-communication norm, a single pipeline diagram (calibrate →
stream → bet → alarm) or an operating-curve plot once real numbers exist
would materially help readability. Not fatal, but flag now so it's not a
last-minute scramble.

## Editorial Decision: Major Revision (interim — full re-adjudication required once Results is real)

Fix now, this round: deanonymization (repository self-reference), abstract's
spatial-claim overclaim, statistical-rigor caveats (small n, small
calibration set, paired-comparison framing), and the missing figure (add a
pipeline diagram, feasible without waiting on training). Do NOT attempt to
resolve the Devil's Advocate CRITICAL this round — it can only be resolved
once the corrected `real_operating_curve_run.json` exists and the
Results/Discussion/Conclusion sections are written against it, honestly, in
whichever direction the numbers point.
