"""
BEV target generation: Gaussian-splatted heatmaps and dense box targets.

Replaces the previous single-pixel target convention
(`heatmap[class_id, row, col] = 1.0`, box regression written only at that
one cell), which was unlearnable in practice: a target that is >99.99%
zero, with each object occupying exactly one of ~16k cells, drives both
heads to a constant. See `craf_x/utils/losses.py` for the matching loss
side of the fix; the two changes are designed together, since the
penalty-reduced focal loss's (1 - y)^beta term is only meaningful against
soft Gaussian targets.

Two properties are preserved deliberately, so the change is a strict
generalization of the old convention rather than a replacement:

- the heatmap peak is still exactly 1.0 at an object's center cell;
- the box regression target at the center cell is bit-identical to what the
  old code wrote there.

What is added is support around the center: the heatmap decays as a 2D
Gaussian whose radius follows the object's footprint (CenterNet's
`gaussian_radius`), and the box target is splatted densely across the cells
inside that Gaussian's core, each carrying the offset from *its own* cell to
the object center. Dense box targets matter here beyond trainability: the
conformal monitor selects its matched cells with a 0.5 threshold on the
heatmap, so every cell above that threshold must carry a valid box target
or the nonconformity score would compare predictions against zeros.
"""
from typing import Sequence, Tuple

import numpy as np
import torch

# Cells whose Gaussian value reaches this level receive a dense box target
# and are supervised by the masked regression loss.
#
# This is deliberately looser than the 0.5 threshold the conformal monitor
# uses to pick its matched cells. With CenterNet's standard sigma =
# (2r+1)/6, the 0.5 contour of a typical vehicle-sized object is a single
# cell, so a 0.5 core would supervise the box head at one cell per object
# -- workable in principle (that is stock CenterNet) but needlessly sparse
# for the short training budget here. At 0.1 an object supervises roughly
# its 3x3 neighbourhood instead.
#
# The two thresholds nest the right way round: the monitor's 0.5 cells are
# a strict subset of these, so every cell the monitor can select is
# guaranteed to carry a valid box target rather than a leftover zero.
REGRESSION_CORE_THRESHOLD = 0.1

# The threshold `conformal_monitor.evaluate.match_mask_from_heatmap` applies
# to select matched cells. Recorded here only so the nesting property above
# is checkable in one place; nothing in craf_x reads it.
MONITOR_MATCH_THRESHOLD = 0.5

# Floor on the Gaussian radius, following CenterPoint's `min_radius`. The
# CornerNet radius formula is expressed in output-grid cells and returns
# sub-1 values for objects that are small relative to the grid: at this
# paper's resolution (128x128 cells over a 100m range, ~0.78 m/cell) a 6m
# vehicle yields radius 0.64, which truncates to 0 and collapses the target
# back to a single pixel. Without this floor the Gaussian splat would be a
# no-op for most real objects in both datasets.
MIN_GAUSSIAN_RADIUS = 2


def gaussian_radius(det_size: Tuple[float, float], min_overlap: float = 0.7) -> float:
    """
    CenterNet/CornerNet Gaussian radius: the largest radius such that a
    predicted box displaced by that much still has at least `min_overlap`
    IoU with the ground-truth box.

    Args:
        det_size: (height, width) of the object's BEV footprint, in cells.
        min_overlap: IoU floor used to derive the radius.

    Returns:
        Radius in cells (>= 0).
    """
    height, width = det_size

    a1 = 1.0
    b1 = height + width
    c1 = width * height * (1.0 - min_overlap) / (1.0 + min_overlap)
    sq1 = np.sqrt(max(b1 ** 2 - 4 * a1 * c1, 0.0))
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4.0
    b2 = 2.0 * (height + width)
    c2 = (1.0 - min_overlap) * width * height
    sq2 = np.sqrt(max(b2 ** 2 - 4 * a2 * c2, 0.0))
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4.0 * min_overlap
    b3 = -2.0 * min_overlap * (height + width)
    c3 = (min_overlap - 1.0) * width * height
    sq3 = np.sqrt(max(b3 ** 2 - 4 * a3 * c3, 0.0))
    r3 = (b3 + sq3) / (2 * a3)

    return float(max(min(r1, r2, r3), 0.0))


def draw_gaussian(heatmap: torch.Tensor, center: Tuple[int, int], radius: int) -> torch.Tensor:
    """
    Max-combine a 2D Gaussian centered at `center` into a single-class
    (H, W) heatmap, in place. Peak value is exactly 1.0 at the center, so
    overlapping objects keep a clean 1.0 peak each.

    Args:
        heatmap: (H, W) tensor, modified in place.
        center: (row, col) center cell.
        radius: Gaussian radius in cells.
    """
    radius = max(int(radius), 0)
    diameter = 2 * radius + 1
    sigma = diameter / 6.0

    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
    gaussian = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2.0 * sigma ** 2))
    # Kill numerical dust so the target stays genuinely sparse away from
    # objects (the focal loss's negative term is sensitive to this).
    gaussian[gaussian < torch.finfo(torch.float32).eps * gaussian.max()] = 0.0

    row, col = center
    height, width = heatmap.shape

    top, bottom = min(row, radius), min(height - row, radius + 1)
    left, right = min(col, radius), min(width - col, radius + 1)
    if bottom <= -top or right <= -left:
        return heatmap

    masked_heatmap = heatmap[row - top: row + bottom, col - left: col + right]
    masked_gaussian = gaussian[radius - top: radius + bottom, radius - left: radius + right]
    torch.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)
    return heatmap


def splat_object(
    heatmap: torch.Tensor,
    regression: torch.Tensor,
    class_id: int,
    center_cell: Tuple[int, int],
    center_fractional: Tuple[float, float],
    box_params: Sequence[float],
    footprint_cells: Tuple[float, float],
    min_overlap: float = 0.7,
    min_radius: int = MIN_GAUSSIAN_RADIUS,
) -> None:
    """
    Write one object's heatmap and dense box-regression targets, in place.

    Args:
        heatmap: (num_classes, H, W) target heatmap.
        regression: (6, H, W) target box regression.
        class_id: object class channel.
        center_cell: (row, col) of the object center.
        center_fractional: the object center in continuous cell coordinates,
            (row_f, col_f) — so `row_f - row` is the sub-cell offset the old
            single-pixel convention stored as `dx`.
        box_params: the four non-offset regression values
            (z_norm, d1, d2, d3), written unchanged at every core cell.
        footprint_cells: the object's (height, width) footprint in cells,
            used to derive the Gaussian radius.
        min_overlap: IoU floor for the radius derivation.
        min_radius: floor on the derived radius (see MIN_GAUSSIAN_RADIUS).
    """
    row, col = center_cell
    row_f, col_f = center_fractional
    radius = max(int(gaussian_radius(footprint_cells, min_overlap)), int(min_radius))

    draw_gaussian(heatmap[class_id], (row, col), radius)

    # Dense box targets over the Gaussian's core, each cell carrying the
    # offset from itself to the true object center. At the center cell this
    # reproduces the old single-pixel target exactly.
    height, width = heatmap.shape[-2:]
    sigma = (2 * radius + 1) / 6.0
    for r in range(max(row - radius, 0), min(row + radius + 1, height)):
        for c in range(max(col - radius, 0), min(col + radius + 1, width)):
            if radius > 0:
                value = float(np.exp(-(((c - col) ** 2 + (r - row) ** 2) / (2.0 * sigma ** 2))))
                if value < REGRESSION_CORE_THRESHOLD:
                    continue
            regression[0, r, c] = row_f - r
            regression[1, r, c] = col_f - c
            regression[2, r, c] = box_params[0]
            regression[3, r, c] = box_params[1]
            regression[4, r, c] = box_params[2]
            regression[5, r, c] = box_params[3]


def object_center_mask(
    heatmap: torch.Tensor, threshold: float = REGRESSION_CORE_THRESHOLD
) -> torch.Tensor:
    """
    (1, H, W) binary mask of cells belonging to some object's Gaussian core
    -- the cells that carry a valid dense box target and are therefore the
    ones the regression loss is masked to.

    Accepts either a (num_classes, H, W) target or a batched
    (B, num_classes, H, W) target, returning (1, H, W) or (B, 1, H, W).
    """
    peak = heatmap.amax(dim=-3, keepdim=True)
    return (peak >= threshold).float()
