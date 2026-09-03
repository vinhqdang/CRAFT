# Papers

This directory holds one subdirectory per research paper built on the shared [`craf_x`](../craf_x) library. Each paper owns its manuscript, planning notes, reviewer correspondence, and any paper-specific figure/experiment scripts; none of that belongs in the shared library or its tests.

| Directory | Paper | Status |
|---|---|---|
| [`craf-x-tvc/`](craf-x-tvc/) | CRAF-X: Cross-modal Robust Adaptive Fusion with eXplainability | Under review at *The Visual Computer* |
| [`conformal-snow-icra2027/`](conformal-snow-icra2027/) | Anytime-Valid Conformal Monitoring for Weather-Onset Detection | Prototype implemented, not yet evaluated — targeting ICRA 2027 |

## Adding a new paper

1. Create `papers/<paper-slug>/` (short, kebab-case, e.g. `craf-x-tvc`).
2. Put that paper's `manuscript/`, planning docs, and reviewer correspondence inside it.
3. If the paper needs code beyond what `craf_x/` already provides, extend `craf_x/` (if the addition is broadly reusable) or add a `papers/<paper-slug>/scripts/` or `experiments/` directory for paper-specific glue code, following the pattern in `craf-x-tvc/scripts/`.
4. Add a `README.md` in the paper's directory describing its abstract, status, and citation, and list it in the table above.
