"""
Signal-strength improvements to the conformal monitoring layer.

The three prior experiments on this paper's monitor (the CCP-informed
bettor, its snow-corruption-retrained variant, and mixture betting over
covariates) all modified the *betting rule* -- how aggressively to stake on
the evidence. None improved detection delay over the covariate-blind
baseline. Detection delay, however, is dominated by the *evidence itself*:
how far the frame-level miscoverage rate m(t) actually moves at onset. A
weak m(t) jump cannot be rescued by any betting rule.

This package attacks the evidence instead, in two independent ways:

- `phantom_score`: the existing nonconformity score is the box-regression
  residual restricted to cells a ground-truth object occupies, which is
  structurally blind to snow's dominant physical effect (spurious
  near-range LiDAR returns producing phantom detections in cells with *no*
  ground-truth object). Adds a phantom-detection component.
- `evalue_merge`: the existing global monitor averages miscoverage across
  the whole BEV grid *before* betting, diluting localized degradation.
  Merges per-cell e-values *after* each cell has amplified its own local
  signal instead.

Both reuse `conformal_monitor` (read-only) for the underlying calibration,
betting and spatial machinery.
"""
