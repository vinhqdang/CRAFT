# VP-NAV (Image and Vision Computing): From Robust Fusion to Statistically Valid Monitoring

**Status:** First draft complete, not yet submitted. Targeting the *Image and Vision
Computing* (Elsevier) special issue "Visual Perception enabling Autonomous Navigation
(VP-NAV)", deadline 31 December 2026.

This paper presents a statistically principled anytime-valid conformal monitoring layer
for multimodal 3D autonomous-vehicle perception: a monitored camera/LiDAR fusion
detector is described generically as background infrastructure (its verify-then-fuse
design — a cross-modal consistency probe, a gated fusion module, an adversarial
consistency training objective), and the paper's own contribution is the monitoring
layer built on top of it — spatially-resolved anytime-valid sequential testing with
online false-discovery-rate control, evaluated against real adverse-weather driving
data (Snowy Scenes, and a second independent dataset, CADC, evaluated concurrently with
this draft).

**Read [`plan.md`](plan.md) before submitting.** It records the venue research (format,
open items still needing confirmation — anonymity policy, length guidance) and, for
internal record-keeping only (this language does not appear in the manuscript or cover
letter), the reasoning behind why the detector is described generically rather than
attributed to a specific external paper.

## Contents

- `manuscript/` — LaTeX source (`elsarticle` class) and compiled PDF.
- `plan.md` — venue research notes and internal disclosure/differentiation record.
- `cover_letter.txt` — draft cover letter.

## Relationship to this repository's other work

- The monitored detector's implementation lives in the shared [`craf_x`](../../craf_x)
  library, as it does for this repository's other papers; this paper's own manuscript
  describes that detector generically rather than naming or citing its own paper
  (see `plan.md`).
- Extends [`conformal-snow-icra2027/`](../conformal-snow-icra2027/) with a second real
  dataset (CADC) and journal-length depth; the ICRA paper's own submitted PDF is
  untouched by this work.
