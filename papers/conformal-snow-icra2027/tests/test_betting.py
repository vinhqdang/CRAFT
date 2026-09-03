from conformal_monitor.betting import (
    AGRAPABettor,
    CCPInformedBettor,
    SequentialTester,
    SFOGDBettor,
    WealthProcess,
)


def test_wealth_process_stays_flat_under_zero_bet():
    wp = WealthProcess(alpha=0.1)
    wp.step(m_t=0.9, lambda_t=0.0)  # lambda=0 -> factor=1 regardless of m_t
    wp.step(m_t=0.0, lambda_t=0.0)
    assert wp.wealth == 1.0


def test_wealth_process_grows_when_betting_on_true_violation():
    wp = WealthProcess(alpha=0.1, lambda_max=5.0)
    # m_t=1.0 consistently (worst case) with a fixed positive bet should grow wealth
    for _ in range(10):
        wp.step(m_t=1.0, lambda_t=1.0)
    assert wp.wealth > 1.0


def test_wealth_process_clips_lambda_to_bounds():
    wp = WealthProcess(alpha=0.1, lambda_max=2.0)
    wp.step(m_t=1.0, lambda_t=100.0)  # should clip to lambda_max=2.0
    assert wp.wealth == 1.0 + 2.0 * (1.0 - 0.1)


def test_agrapa_bettor_lambda_within_bounds():
    bettor = AGRAPABettor(alpha=0.1)
    for m_t in [0.5, 0.9, 0.0, 0.3, 1.0]:
        lam = bettor.next_lambda()
        assert 0.0 <= lam <= bettor.lambda_max
        bettor.update(m_t)


def test_sfogd_bettor_lambda_within_bounds():
    bettor = SFOGDBettor(alpha=0.1)
    for m_t in [0.5, 0.9, 0.0, 0.3, 1.0]:
        lam = bettor.next_lambda()
        assert 0.0 <= lam <= bettor.lambda_max
        bettor.update(m_t)


class _StubBettor:
    """Deterministic base bettor so CCPInformedBettor's scaling is checkable."""

    def __init__(self, alpha):
        self.lambda_max = 1.0 / (1.0 - alpha)
        self._fixed_lambda = 0.2

    def next_lambda(self, **_):
        return self._fixed_lambda

    def update(self, m_t, **_):
        pass


def test_ccp_informed_bettor_scales_up_with_disagreement():
    base = _StubBettor(alpha=0.1)
    ccp_bettor = CCPInformedBettor(base, kappa=1.0)

    lam_no_disagreement = ccp_bettor.next_lambda(ccp_disagreement=0.0)
    lam_with_disagreement = ccp_bettor.next_lambda(ccp_disagreement=1.0)

    assert lam_no_disagreement == base._fixed_lambda
    assert lam_with_disagreement == base._fixed_lambda * 2.0
    assert lam_with_disagreement > lam_no_disagreement


def test_ccp_informed_bettor_clips_to_lambda_max():
    base = _StubBettor(alpha=0.1)
    base._fixed_lambda = base.lambda_max  # already at the cap
    ccp_bettor = CCPInformedBettor(base, kappa=5.0)

    lam = ccp_bettor.next_lambda(ccp_disagreement=1.0)
    assert lam == ccp_bettor.lambda_max


def test_sequential_tester_alarms_on_persistent_violation():
    alpha, delta = 0.1, 0.05
    tester = SequentialTester(alpha, delta, AGRAPABettor(alpha))

    m_stream = [1.0] * 200  # persistent, maximal violation of E[m] <= alpha
    alarm_time = tester.run(m_stream)

    assert alarm_time is not None
    assert alarm_time < 200


def test_sequential_tester_does_not_alarm_under_null():
    alpha, delta = 0.3, 0.01
    tester = SequentialTester(alpha, delta, AGRAPABettor(alpha))

    # m_t exactly at the null boundary forever: E[m_t - alpha] = 0
    m_stream = [alpha] * 200
    alarm_time = tester.run(m_stream)

    assert alarm_time is None
