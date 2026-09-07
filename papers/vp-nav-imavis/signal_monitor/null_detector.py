"""
Null-detector control: what does the monitor achieve with no functioning
detector at all?

The nonconformity score ||B_pred - B_target||_1 degenerates to
||B_target||_1 when the box head contributes nothing, so a monitor running
on a null detector is measuring purely the ground-truth scene-content
difference between the nominal and degraded halves of the onset stream. Any
detection delay it achieves is attributable to the splice, not to
perception degradation.

That matters here because the two datasets' content artifacts point in
opposite directions. Measured as the nominal-to-degraded jump in m(t) under
a zeroed box head: Snowy Scenes is negative (-0.216) while its real
detector's jump is positive, so content works *against* detection there and
a positive result cannot be content-driven. CADC's is positive (+0.139),
the *same* direction as its real detector's (+0.065) -- so on CADC, real
degradation signal and scene-content difference are confounded by sign, and
a good detection number could be riding on either.

Running the full operating curve through a null detector settles it
directly, and is worth keeping as a permanent row in the results rather
than a one-off check: "our monitor versus a monitor with no detector" is
the cheapest convincing control available, and this paper has already been
burned once by exactly this failure mode.

Implementation note: rather than duplicating the evaluation pipeline with a
zeroing flag threaded through it, this wraps the model so that every
existing code path -- calibration included -- sees a detector whose box
head outputs zeros. The control therefore exercises the same
`conformal_monitor.evaluate` functions as the real result, with no
parallel implementation that could drift from it.
"""
import torch
import torch.nn as nn


class NullBoxHeadModel(nn.Module):
    """
    Wraps a detector so its box-regression output is identically zero,
    leaving every other output untouched.

    Calibration must run through this wrapper too: a null detector
    calibrates its own quantile on its own scores, which is what a
    deployment with a broken box head would actually do. Passing the
    wrapper to `calibrate_on_clear_weather` and to the operating-curve
    functions achieves that without either of them needing to know.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, pointcloud: torch.Tensor):
        out = dict(self.model(image, pointcloud))
        out["B"] = torch.zeros_like(out["B"])
        return out

    def eval(self):
        self.model.eval()
        return super().eval()

    def train(self, mode: bool = True):
        self.model.train(mode)
        return super().train(mode)
