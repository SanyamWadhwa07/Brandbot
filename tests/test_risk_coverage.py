import numpy as np

from brandbot.eval import risk_coverage as rc


def test_a_perfect_signal_beats_a_useless_one():
    failed = np.array([True] * 40 + [False] * 60)
    perfect = np.where(failed, 0.0, 1.0)  # confident exactly when correct
    useless = np.full(100, 0.5)
    assert rc.curve(perfect, failed).aurc < rc.curve(useless, failed).aurc


def test_escalating_everything_carries_no_risk_and_no_coverage():
    failed = np.array([True, False] * 20)
    top = max(rc.curve(np.linspace(0, 1, 40), failed).points, key=lambda p: p.threshold)
    assert top.coverage == 0.0
    assert top.risk == 0.0


def test_content_rules_override_high_confidence():
    failed = np.array([True] * 10 + [False] * 10)
    confident = np.ones(20)
    forced = np.array([True] * 10 + [False] * 10)  # the failures are all rule-hits
    point = rc.curve(confident, failed, must_escalate=forced).points[0]
    assert point.coverage == 0.5
    assert point.risk == 0.0


def test_raising_the_cost_of_a_bad_reply_lowers_coverage():
    rng = np.random.default_rng(0)
    conf = rng.random(200)
    failed = rng.random(200) > conf  # failure gets likelier as confidence drops
    cheap = rc.curve(conf, failed, cost_ratio=1.0).at_cost_ratio(1.0)
    dear = rc.curve(conf, failed, cost_ratio=20.0).at_cost_ratio(20.0)
    assert dear.coverage <= cheap.coverage


def test_risk_is_measured_over_sent_replies_not_all_replies():
    failed = np.array([True, False, False, False])
    conf = np.array([0.9, 0.1, 0.1, 0.1])
    point = next(p for p in rc.curve(conf, failed).points if p.threshold == 0.9)
    assert point.coverage == 0.25
    assert point.risk == 1.0  # the single reply we sent was bad
