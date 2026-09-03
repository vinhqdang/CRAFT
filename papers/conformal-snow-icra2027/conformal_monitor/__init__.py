"""
Anytime-valid conformal monitoring for weather-onset detection.

Paper-local implementation for `papers/conformal-snow-icra2027/` (see plan.md).
Wraps craf_x's detector outputs (and, for the covariate-informed bettor, its
Cross-modal Consistency Probe score) in split conformal prediction plus a
sequential e-process (testing-by-betting) to flag the onset of distribution
shift with anytime-valid Type-I error control.
"""
