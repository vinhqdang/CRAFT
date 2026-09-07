import numpy as np
import pytest
import torch

from craf_x.utils.targets import (
    MONITOR_MATCH_THRESHOLD,
    REGRESSION_CORE_THRESHOLD,
    draw_gaussian,
    gaussian_radius,
    object_center_mask,
    splat_object,
)


def test_gaussian_radius_is_non_negative_and_grows_with_object_size():
    small = gaussian_radius((2.0, 2.0))
    large = gaussian_radius((20.0, 20.0))
    assert small >= 0.0
    assert large > small


def test_gaussian_radius_handles_degenerate_footprint():
    # A zero-area footprint must not produce a NaN or a negative radius.
    r = gaussian_radius((0.0, 0.0))
    assert r == pytest.approx(0.0)
    assert not np.isnan(r)


def test_draw_gaussian_peaks_at_exactly_one_at_the_center():
    heatmap = torch.zeros(16, 16)
    draw_gaussian(heatmap, (8, 8), radius=3)

    assert heatmap[8, 8] == pytest.approx(1.0)
    assert heatmap.max() == pytest.approx(1.0)
    # Decays away from the center, and stays zero well outside the radius.
    assert heatmap[8, 9] < heatmap[8, 8]
    assert heatmap[8, 14] == 0.0


def test_draw_gaussian_clips_at_array_edges_without_error():
    heatmap = torch.zeros(8, 8)
    draw_gaussian(heatmap, (0, 0), radius=4)  # center hard against the corner

    assert heatmap[0, 0] == pytest.approx(1.0)
    assert torch.isfinite(heatmap).all()


def test_draw_gaussian_max_combines_overlapping_objects():
    heatmap = torch.zeros(16, 16)
    draw_gaussian(heatmap, (8, 8), radius=3)
    draw_gaussian(heatmap, (8, 10), radius=3)

    # Both centers keep a clean 1.0 peak rather than summing past it.
    assert heatmap[8, 8] == pytest.approx(1.0)
    assert heatmap[8, 10] == pytest.approx(1.0)
    assert heatmap.max() == pytest.approx(1.0)


def test_splat_object_reproduces_the_old_single_pixel_target_at_the_center():
    # The key backward-compatibility guard: whatever else splatting adds,
    # the center cell must carry exactly what the previous
    # `heatmap[c, row, col] = 1.0` / `regression[:, row, col] = ...`
    # convention wrote there.
    heatmap = torch.zeros(3, 32, 32)
    regression = torch.zeros(6, 32, 32)
    row, col = 10, 12
    dx, dy = 0.25, 0.75  # sub-cell offsets, as the old code computed them
    z_norm, d1, d2, d3 = 0.4, 1.8, 4.2, 1.5

    splat_object(
        heatmap, regression,
        class_id=1,
        center_cell=(row, col),
        center_fractional=(row + dx, col + dy),
        box_params=(z_norm, d1, d2, d3),
        footprint_cells=(4.0, 2.0),
    )

    assert heatmap[1, row, col] == pytest.approx(1.0)
    assert regression[0, row, col] == pytest.approx(dx)
    assert regression[1, row, col] == pytest.approx(dy)
    assert regression[2, row, col] == pytest.approx(z_norm)
    assert regression[3, row, col] == pytest.approx(d1)
    assert regression[4, row, col] == pytest.approx(d2)
    assert regression[5, row, col] == pytest.approx(d3)


def test_splat_object_writes_dense_targets_over_the_gaussian_core():
    # Every cell the monitor would select (heatmap >= 0.5) must carry a real
    # box target -- otherwise the nonconformity score compares predictions
    # against zeros on the Gaussian shoulder.
    heatmap = torch.zeros(3, 32, 32)
    regression = torch.zeros(6, 32, 32)
    splat_object(
        heatmap, regression,
        class_id=0,
        center_cell=(16, 16),
        center_fractional=(16.5, 16.5),
        box_params=(0.3, 2.0, 4.0, 1.6),
        footprint_cells=(8.0, 6.0),
    )

    core = object_center_mask(heatmap)[0].bool()
    assert core.sum() > 1  # genuinely dense, not a single pixel
    # Dimensions are constant across the core; every core cell is supervised.
    assert regression[3][core].tolist() == pytest.approx([2.0] * int(core.sum()))
    assert not torch.allclose(regression[:2][:, core], torch.zeros_like(regression[:2][:, core]))


def test_splat_object_offsets_point_at_the_true_center_from_each_core_cell():
    heatmap = torch.zeros(1, 32, 32)
    regression = torch.zeros(6, 32, 32)
    row_f, col_f = 16.5, 16.5
    splat_object(
        heatmap, regression,
        class_id=0,
        center_cell=(16, 16),
        center_fractional=(row_f, col_f),
        box_params=(0.0, 1.0, 1.0, 1.0),
        footprint_cells=(8.0, 8.0),
    )

    core = object_center_mask(heatmap)[0].bool()
    rows, cols = torch.nonzero(core, as_tuple=True)
    for r, c in zip(rows.tolist(), cols.tolist()):
        # Each cell's stored offset, added back to the cell index, must
        # recover the one true object center.
        assert regression[0, r, c].item() + r == pytest.approx(row_f)
        assert regression[1, r, c].item() + c == pytest.approx(col_f)


def test_object_center_mask_matches_the_core_threshold():
    heatmap = torch.zeros(2, 16, 16)
    draw_gaussian(heatmap[0], (8, 8), radius=3)

    mask = object_center_mask(heatmap)
    assert mask.shape == (1, 16, 16)
    expected = (heatmap.amax(dim=0, keepdim=True) >= REGRESSION_CORE_THRESHOLD).float()
    assert torch.equal(mask, expected)


def test_object_center_mask_supports_batched_targets():
    heatmap = torch.zeros(4, 2, 16, 16)
    draw_gaussian(heatmap[0, 0], (8, 8), radius=2)

    mask = object_center_mask(heatmap)
    assert mask.shape == (4, 1, 16, 16)
    assert mask[0].sum() > 0
    assert mask[1].sum() == 0  # no object splatted into this frame


def test_monitor_matched_cells_are_a_subset_of_supervised_cells():
    # The property the two thresholds exist to guarantee: every cell the
    # conformal monitor can select (heatmap >= 0.5) must carry a real box
    # target, i.e. lie inside the supervised regression core. Otherwise the
    # nonconformity score would silently compare a prediction against zero.
    heatmap = torch.zeros(2, 48, 48)
    regression = torch.zeros(6, 48, 48)
    for center, footprint in (((12, 12), (8.0, 6.0)), ((30, 33), (2.0, 2.0)), ((40, 8), (20.0, 14.0))):
        splat_object(
            heatmap, regression,
            class_id=0,
            center_cell=center,
            center_fractional=(center[0] + 0.3, center[1] + 0.6),
            box_params=(0.2, 1.9, 4.4, 1.7),
            footprint_cells=footprint,
        )

    monitor_cells = heatmap.amax(dim=0) >= MONITOR_MATCH_THRESHOLD
    supervised = heatmap.amax(dim=0) >= REGRESSION_CORE_THRESHOLD
    assert monitor_cells.sum() > 0
    assert bool((~supervised[monitor_cells]).sum() == 0)
    # And every monitor-selected cell really does carry a non-trivial target.
    assert bool((regression[3][monitor_cells] > 0).all())
