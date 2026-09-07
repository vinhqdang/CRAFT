import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import torch

from craf_x.config import CRAFXConfig
from craf_x.datasets.nuscenes_mock import NuScenesMockDataset
from craf_x.models.crafx_net import CRAFX_Net

from conformal_monitor.evaluate import calibrate_on_clear_weather

from signal_monitor.null_detector import NullBoxHeadModel

CONFORMAL_ALPHA = 0.2


def _build_model():
    return CRAFX_Net(CRAFXConfig(bev_h=32, bev_w=32))


def test_null_detector_zeroes_only_the_box_head():
    model = _build_model().eval()
    wrapped = NullBoxHeadModel(model).eval()
    image = torch.randn(1, 3, 32, 32)
    pointcloud = torch.randn(1, 4, 32, 32)

    with torch.no_grad():
        real = model(image, pointcloud)
        null = wrapped(image, pointcloud)

    assert torch.all(null["B"] == 0.0)
    # Everything else must pass through untouched -- the control isolates the
    # box head, and would prove nothing if it also disturbed the CCP score
    # or the heatmap.
    for key in ("H", "S", "V"):
        assert torch.equal(null[key], real[key]), key


def test_null_detector_does_not_mutate_the_wrapped_model_output():
    # The wrapper copies the dict before zeroing; if it mutated in place, a
    # caller holding the inner model's output would silently see zeros too.
    model = _build_model().eval()
    wrapped = NullBoxHeadModel(model).eval()
    image = torch.randn(1, 3, 32, 32)
    pointcloud = torch.randn(1, 4, 32, 32)

    with torch.no_grad():
        wrapped(image, pointcloud)
        real_after = model(image, pointcloud)

    assert not torch.all(real_after["B"] == 0.0)


def test_null_detector_exposes_parameters_for_device_lookup():
    # conformal_monitor.evaluate finds the device via
    # next(model.parameters()); the wrapper must keep that working.
    wrapped = NullBoxHeadModel(_build_model())
    assert next(wrapped.parameters()) is not None


def test_null_detector_calibrates_through_the_existing_pipeline():
    # The control has to run through the same calibration path as the real
    # result, not a parallel implementation that could drift from it.
    wrapped = NullBoxHeadModel(_build_model()).eval()
    dataset = NuScenesMockDataset(num_samples=4)

    q_hat = calibrate_on_clear_weather(wrapped, dataset, CONFORMAL_ALPHA, batch_size=2)
    assert q_hat >= 0.0


def test_null_detector_quantile_matches_hand_zeroed_scores():
    # The null quantile must equal what you get by calibrating on
    # ||0 - B_target||_1 directly -- i.e. the wrapper really does reduce the
    # score to the ground-truth box magnitude.
    import numpy as np
    from conformal_monitor.calibration import calibrate_quantile, object_nonconformity_scores
    from conformal_monitor.evaluate import match_mask_from_heatmap

    model = _build_model().eval()
    wrapped = NullBoxHeadModel(model).eval()
    dataset = NuScenesMockDataset(num_samples=4)

    torch.manual_seed(3)
    q_wrapped = calibrate_on_clear_weather(wrapped, dataset, CONFORMAL_ALPHA, batch_size=2)

    torch.manual_seed(3)
    scores = []
    for i in range(len(dataset)):
        sample = dataset[i]
        targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
        mask = match_mask_from_heatmap(targets["H"])
        scores.append(
            object_nonconformity_scores(torch.zeros_like(targets["B"]), targets["B"], mask)
        )
    q_direct = calibrate_quantile(np.concatenate(scores), CONFORMAL_ALPHA)

    assert q_wrapped == pytest.approx(q_direct)
