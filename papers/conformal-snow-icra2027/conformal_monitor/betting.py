"""
Sequential testing-by-betting over the conformal miscoverage rate m(t).

Implements the e-process backbone from plan.md ("The e-process (testing by
betting)"), which is the established part of the method, plus:

- Two covariate-blind betting rules (aGRAPA, SF-OGD) that reproduce
  Monroy Muñoz, Verma & Timans (WACV 2026, arXiv:2602.12983) — the baseline
  to beat, per plan.md's "Baselines to beat".
- `CCPInformedBettor`, the paper's primary contribution (plan.md claim 1):
  a betting rule conditioned on craf_x's Cross-modal Consistency Probe (CCP)
  score, an external covariate observed at the same timestep, rather than
  only the history of m(t) itself.

Under H0: E[m(t) | F_{t-1}] <= alpha for all t, the wealth process
K_t = prod_{s<=t} (1 + lambda_s * (m(s) - alpha)) is a nonnegative
supermartingale for any predictable lambda_s in [0, 1/alpha], so by Ville's
inequality P_H0(sup_t K_t >= 1/delta) <= delta. Alarm the first time
K_t >= 1/delta.

The 1/alpha bound (not 1/(1-alpha), which only bounds the m(t)=1 side) is
what's needed: since 1 + lambda*(m(t)-alpha) is linear and increasing in
m(t) in [0, 1], its minimum over that range is at m(t)=0, i.e.
1 - lambda*alpha, which is >= 0 iff lambda <= 1/alpha.
"""
import math
from typing import Optional

import numpy as np


class WealthProcess:
    """Tracks K_t = prod (1 + lambda_s * (m(s) - alpha)) and the alarm rule."""

    def __init__(self, alpha: float, lambda_max: Optional[float] = None):
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.lambda_max = lambda_max if lambda_max is not None else 1.0 / alpha
        self.wealth = 1.0
        self.history = []  # list of (m_t, lambda_t, wealth_t)

    def step(self, m_t: float, lambda_t: float) -> float:
        lambda_t = float(np.clip(lambda_t, 0.0, self.lambda_max))
        factor = 1.0 + lambda_t * (m_t - self.alpha)
        # Guard against floating-point noise only: factor is theoretically
        # >= 0 whenever lambda_t <= 1/alpha (see module docstring).
        factor = max(factor, 0.0)
        self.wealth *= factor
        self.history.append((m_t, lambda_t, self.wealth))
        return self.wealth

    def alarmed(self, delta: float) -> bool:
        return self.wealth >= 1.0 / delta


class _RunningMoments:
    """Running mean/variance of the centered variable X_t = m(t) - alpha."""

    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.mean_sq = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.mean_sq += (x * x - self.mean_sq) / self.n

    @property
    def variance(self) -> float:
        return max(self.mean_sq - self.mean ** 2, 0.0)


class AGRAPABettor:
    """
    Approximate GRAPA (Growth-Rate Adaptive to Predictable Averages),
    Waudby-Smith & Ramdas (2020), applied to X_t = m(t) - alpha.

    lambda_t = clip(mean(X_{<t}) / (E[X_{<t}^2] + eps), 0, lambda_max).
    Covariate-blind: uses only the history of m(t) itself. This is the
    baseline referenced in plan.md item 1 ("Baselines to beat").
    """

    def __init__(self, alpha: float, lambda_max: Optional[float] = None, eps: float = 1e-6):
        self.alpha = alpha
        self.lambda_max = lambda_max if lambda_max is not None else 1.0 / alpha
        self.eps = eps
        self._moments = _RunningMoments()

    def next_lambda(self, **_covariates) -> float:
        if self._moments.n == 0:
            return 0.0
        second_moment = self._moments.variance + self._moments.mean ** 2
        lam = self._moments.mean / (second_moment + self.eps)
        return float(np.clip(lam, 0.0, self.lambda_max))

    def update(self, m_t: float, **_covariates) -> None:
        self._moments.update(m_t - self.alpha)


class SFOGDBettor:
    """
    Scale-Free Online Gradient Descent betting (AdaGrad-style step size on
    the log-wealth gradient), also from the sequential-betting literature and
    used as an alternative covariate-blind baseline in Monroy Muñoz et al.

    lambda_{t+1} = clip(lambda_t + eta_t * grad_t, 0, lambda_max),
    grad_t = X_t / (1 + lambda_t * X_t),  eta_t = D / sqrt(1 + sum_{s<=t} grad_s^2)
    """

    def __init__(self, alpha: float, lambda_max: Optional[float] = None):
        self.alpha = alpha
        self.lambda_max = lambda_max if lambda_max is not None else 1.0 / alpha
        self._lambda = 0.0
        self._sq_grad_sum = 0.0

    def next_lambda(self, **_covariates) -> float:
        return float(np.clip(self._lambda, 0.0, self.lambda_max))

    def update(self, m_t: float, **_covariates) -> None:
        lam = self.next_lambda()
        x_t = m_t - self.alpha
        denom = 1.0 + lam * x_t
        grad = x_t / denom if abs(denom) > 1e-8 else 0.0
        self._sq_grad_sum += grad ** 2
        eta = self.lambda_max / math.sqrt(1.0 + self._sq_grad_sum)
        self._lambda = float(np.clip(self._lambda + eta * grad, 0.0, self.lambda_max))


class CCPInformedBettor:
    """
    Covariate-informed betting rule (plan.md claim 1, the paper's primary
    contribution): scales a covariate-blind base bettor's stake by the
    current cross-modal disagreement signal, so the process bets more
    aggressively on frames where craf_x's CCP already flags camera/LiDAR
    disagreement — a leading indicator of degradation — rather than reacting
    only to the lagging miscoverage rate m(t) itself.

    lambda_t = clip(lambda_base_t * (1 + kappa * ccp_disagreement_t), 0, lambda_max)

    where ccp_disagreement_t = mean(1 - S_t) over the frame (or cell), S_t
    being craf_x's per-cell consistency score in [0, 1] (see
    craf_x/models/ccp.py) — 0 disagreement leaves the base bet unchanged,
    higher disagreement scales it up.

    Args:
        base_bettor: an AGRAPABettor or SFOGDBettor instance providing the
            covariate-blind component.
        kappa: gain on the covariate term (kappa=0 reduces to base_bettor).
    """

    def __init__(self, base_bettor, kappa: float = 1.0):
        self.base_bettor = base_bettor
        self.kappa = kappa
        self.lambda_max = base_bettor.lambda_max

    def next_lambda(self, ccp_disagreement: float = 0.0, **_covariates) -> float:
        base_lambda = self.base_bettor.next_lambda()
        scaled = base_lambda * (1.0 + self.kappa * ccp_disagreement)
        return float(np.clip(scaled, 0.0, self.lambda_max))

    def update(self, m_t: float, ccp_disagreement: float = 0.0, **_covariates) -> None:
        self.base_bettor.update(m_t)


class SequentialTester:
    """
    Drives a WealthProcess with a given bettor over a stream of
    (m_t, covariates) pairs and reports the anytime-valid alarm time.
    """

    def __init__(self, alpha: float, delta: float, bettor):
        self.alpha = alpha
        self.delta = delta
        self.bettor = bettor
        self.wealth_process = WealthProcess(alpha, lambda_max=bettor.lambda_max)
        self.alarm_time: Optional[int] = None

    def step(self, m_t: float, t: int, **covariates) -> float:
        """Feed one frame's miscoverage rate; returns the updated wealth."""
        lam = self.bettor.next_lambda(**covariates)
        wealth = self.wealth_process.step(m_t, lam)
        self.bettor.update(m_t, **covariates)
        if self.alarm_time is None and self.wealth_process.alarmed(self.delta):
            self.alarm_time = t
        return wealth

    def run(self, m_stream, covariate_stream=None) -> Optional[int]:
        """
        Run over a full stream of miscoverage rates (and optional per-step
        covariate dicts). Returns the first alarm time, or None if the
        process never alarms.
        """
        if covariate_stream is None:
            covariate_stream = ({} for _ in m_stream)
        for t, (m_t, cov) in enumerate(zip(m_stream, covariate_stream)):
            self.step(m_t, t, **cov)
            if self.alarm_time is not None:
                return self.alarm_time
        return None
