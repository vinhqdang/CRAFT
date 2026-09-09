import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts")))

import pytest

from compute_mde import mde_from_ci, Z_SUM, Z_975


def test_mde_scales_linearly_with_ci_width():
    narrow = mde_from_ci(-0.05, 0.05)
    wide = mde_from_ci(-0.10, 0.10)
    assert wide["mde_80pct_power"] == pytest.approx(2 * narrow["mde_80pct_power"])


def test_se_matches_normal_approximation_of_the_ci():
    # A 95% CI of [-0.0649, +0.0649] implies SE ~= 0.0649 / 1.96 ~= 0.0331.
    result = mde_from_ci(-0.0649, 0.0649)
    assert result["se_normal_approx"] == pytest.approx(0.0649 / Z_975, rel=1e-6)


def test_mde_formula_uses_the_documented_z_sum():
    result = mde_from_ci(-0.1, 0.1)
    expected_mde = Z_SUM * result["se_normal_approx"]
    assert result["mde_80pct_power"] == pytest.approx(expected_mde)


def test_asymmetric_ci_uses_half_width_not_full_width():
    # CI need not be symmetric around a point estimate; half-width is what
    # feeds the SE estimate.
    result = mde_from_ci(-0.02, 0.10)
    assert result["ci_half_width"] == pytest.approx(0.06)
