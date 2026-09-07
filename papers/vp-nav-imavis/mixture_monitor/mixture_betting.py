"""
Mixture (hedged) betting: combine multiple candidate bettors' wealth
processes into a single e-process via a convex combination, so no a
priori commitment to any one betting strategy is needed.

Motivation (papers/vp-nav-imavis/manuscript): across three real
experiments in this project, neither the covariate-blind nor the
CCP-informed bettor dominates consistently -- CCP-informed hurt on real
Snowy Scenes data (censored at every delta tested) but tied or slightly
helped on real CADC data (delay 6/7/8 vs. 6/8/9 frames at delta=0.3/0.1/
0.05). There is no way to know in advance, for a new deployment, which
regime applies. Hedging across both bettors at once is the natural
response, and it is provably valid, not just an empirical hope:

Theoretical foundation: Vovk, V. & Wang, R. (2021), "E-values:
Calibration, combination and applications," Annals of Statistics 49(3),
1736-1754 (arXiv:1912.06116). Their central result -- stated directly in
the paper's own abstract -- is that e-values (nonnegative random
variables with expectation at most 1 under the null, which is exactly
what conformal_monitor.betting.WealthProcess produces at every fixed t;
see conformal_monitor/betting.py's own docstring for that derivation) can
be merged simply by averaging them, and any convex combination of valid
e-values is itself a valid e-value. Since each of our per-timestep wealth
values K_i(t) is an e-value at any fixed t (nonnegative, E_H0[K_i(t)] <=
1), a convex combination K_mix(t) = sum_i w_i * K_i(t), with weights w_i
>= 0 summing to 1 chosen without looking at the future, is itself a valid
e-value at every fixed t. Applying Ville's inequality to K_mix(t) exactly
as conformal_monitor.betting.WealthProcess already does for a single
bettor's K_t (see that module's own docstring) then gives the same
anytime-valid false-alarm guarantee for the mixture as for any single
component bettor -- this is what makes hedging across bettors provably
safe rather than merely empirically convenient.

Related but different prior work (verified directly, not taken on faith):
- Prinster, Han, Liu & Saria, "WATCH: Adaptive Monitoring for AI
  Deployments via Weighted-Conformal Martingales," ICML 2025
  (arXiv:2505.04608) -- a weighted generalization of ONE conformal test
  martingale for online adaptation to covariate shift. Different
  mechanism (reweighting a single martingale's own construction) and
  different goal (adapting to mild shift while still detecting severe
  shift) from combining multiple structurally distinct betting
  strategies to hedge against not knowing which one helps.
- Eliades & Papadopoulos, "ICM Ensemble with Novel Betting Functions for
  Concept Drift" (arXiv:2406.15760, 2024) and "A Conformal Martingales
  Ensemble Approach for addressing Concept Drift" (12th Symposium on
  Conformal and Probabilistic Prediction with Applications, PMLR vol.
  204, 2023) -- ensemble multiple DENSITY ESTIMATORS inside one betting
  function for concept-drift detection, not multiple full wealth
  processes representing different covariate strategies.
Neither combines a covariate-blind and a covariate-informed bettor
specifically to hedge against not knowing, in advance, which one will
help in a given real deployment -- the gap this module addresses.

This module deliberately imports conformal_monitor's existing pieces
(WealthProcess, alarm_time_from_trajectory, _to_device,
match_mask_from_heatmap) rather than modifying that package, per this
project's own coordination discipline around shared code used by
concurrent work.
"""
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "conformal-snow-icra2027")))

from conformal_monitor.betting import WealthProcess
from conformal_monitor.calibration import frame_miscoverage_rate, object_nonconformity_scores
from conformal_monitor.corruption import WeatherOnsetStream
from conformal_monitor.evaluate import _to_device, alarm_time_from_trajectory, match_mask_from_heatmap


def compute_mixture_wealth_trajectory(
    model,
    stream,
    q_hat: float,
    alpha: float,
    bettor_factories: Dict[str, Callable[[], object]],
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, List[float]]:
    """
    Runs the model once per frame over `stream` and drives one independent
    WealthProcess per entry in `bettor_factories` in parallel (each fed the
    same m(t) and ccp_disagreement(t) at every frame, so all components see
    identical evidence and differ only in their own betting rule), plus a
    mixture trajectory combining their wealth VALUES (not their lambdas) by
    the given convex weights at every timestep -- the combination Vovk &
    Wang (2021) shows is itself a valid e-value (see module docstring).

    Args:
        bettor_factories: name -> zero-arg constructor for a bettor
            instance (e.g. {"blind": lambda: AGRAPABettor(alpha),
            "ccp": lambda: CCPInformedBettor(AGRAPABettor(alpha), kappa=2.0)}).
        weights: name -> convex weight (must be in bettor_factories' keys,
            non-negative, summing to 1). Defaults to equal weight over all
            components if not given.

    Returns:
        Dict mapping each component name to its own wealth trajectory, plus
        the key "mixture" for the combined trajectory.
    """
    names = list(bettor_factories.keys())
    if weights is None:
        weights = {name: 1.0 / len(names) for name in names}
    else:
        if set(weights.keys()) != set(names):
            raise ValueError("weights must have exactly the same keys as bettor_factories")
        total = sum(weights.values())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"weights must sum to 1, got {total}")
        if any(w < 0 for w in weights.values()):
            raise ValueError("weights must be non-negative")

    model.eval()
    device = next(model.parameters()).device
    bettors = {name: factory() for name, factory in bettor_factories.items()}
    wealth_processes = {
        name: WealthProcess(alpha, lambda_max=bettors[name].lambda_max) for name in names
    }
    trajectories: Dict[str, List[float]] = {name: [] for name in names}
    mixture_trajectory: List[float] = []

    with torch.no_grad():
        for t in range(len(stream)):
            sample = stream[t]
            image = sample["image"].unsqueeze(0)
            pointcloud = sample["pointcloud"].unsqueeze(0)
            targets = {k: v.unsqueeze(0) for k, v in sample["targets"].items()}
            image, pointcloud, targets = _to_device(image, pointcloud, targets, device)

            out = model(image, pointcloud)
            match_mask = match_mask_from_heatmap(targets["H"])
            scores = object_nonconformity_scores(out["B"], targets["B"], match_mask)
            m_t = frame_miscoverage_rate(scores, q_hat)
            ccp_disagreement = float((1.0 - out["S"]).mean().item())

            mixture_wealth = 0.0
            for name in names:
                lam = bettors[name].next_lambda(ccp_disagreement=ccp_disagreement)
                wealth = wealth_processes[name].step(m_t, lam)
                bettors[name].update(m_t, ccp_disagreement=ccp_disagreement)
                trajectories[name].append(wealth)
                mixture_wealth += weights[name] * wealth
            mixture_trajectory.append(mixture_wealth)

    trajectories["mixture"] = mixture_trajectory
    return trajectories


def mixture_operating_curve(
    model,
    q_hat: float,
    alpha: float,
    deltas: Sequence[float],
    onset_stream_factory: Callable[[], "WeatherOnsetStream"],
    clear_stream_factory: Callable[[], "WeatherOnsetStream"],
    bettor_factories: Dict[str, Callable[[], object]],
    weights: Optional[Dict[str, float]] = None,
    n_onset_replicates: int = 5,
    n_clear_replicates: int = 5,
) -> Dict[str, List[dict]]:
    """
    Same structure as conformal_monitor.evaluate.operating_curve, but
    reports one curve per component bettor AND one for the mixture, all
    computed from the same replicate streams (so every component is
    compared on identical evidence, not independently resampled data).
    """
    names = list(bettor_factories.keys()) + ["mixture"]

    onset_runs = []
    for _ in range(n_onset_replicates):
        stream = onset_stream_factory()
        trajectories = compute_mixture_wealth_trajectory(model, stream, q_hat, alpha, bettor_factories, weights)
        onset_runs.append((trajectories, stream.onset_frame))

    clear_trajectories_runs = [
        compute_mixture_wealth_trajectory(model, clear_stream_factory(), q_hat, alpha, bettor_factories, weights)
        for _ in range(n_clear_replicates)
    ]

    curves: Dict[str, List[dict]] = {name: [] for name in names}
    for delta in deltas:
        for name in names:
            delays = []
            n_censored = 0
            for trajectories, onset_frame in onset_runs:
                alarm_time = alarm_time_from_trajectory(trajectories[name], delta)
                if alarm_time is None:
                    n_censored += 1
                else:
                    delays.append(alarm_time - onset_frame)

            n_alarmed = sum(
                1 for trajectories in clear_trajectories_runs
                if alarm_time_from_trajectory(trajectories[name], delta) is not None
            )
            fa_rate = n_alarmed / n_clear_replicates if n_clear_replicates else 0.0

            curves[name].append({
                "delta": delta,
                "false_alarm_rate": fa_rate,
                "mean_detection_delay": float(np.mean(delays)) if delays else None,
                "n_censored": n_censored,
            })
    return curves
