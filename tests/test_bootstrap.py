import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from brandbot.eval import bootstrap


def macro_f1(t, p):
    return f1_score(t, p, average="macro", zero_division=0)


def test_interval_brackets_the_point_estimate():
    truth = np.array([0, 1] * 60)
    pred = truth.copy()
    pred[:12] = 1 - pred[:12]
    out = bootstrap.ci(truth, pred, accuracy_score, resamples=400)
    assert out.low <= out.point <= out.high


def test_smaller_samples_give_wider_intervals():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 2, 800)
    pred = np.where(rng.random(800) < 0.8, truth, 1 - truth)
    wide = bootstrap.ci(truth[:50], pred[:50], accuracy_score, resamples=400)
    narrow = bootstrap.ci(truth, pred, accuracy_score, resamples=400)
    assert (wide.high - wide.low) > (narrow.high - narrow.low)


def test_identical_systems_have_zero_delta_and_no_significance():
    truth = np.array([0, 1, 2] * 40)
    pred = np.roll(truth, 1)
    out = bootstrap.compare(truth, pred, pred, macro_f1, resamples=200)
    assert out.delta.point == 0.0
    assert not out.significant


def test_a_large_real_gap_is_detected():
    truth = np.array([0, 1] * 100)
    good = truth.copy()
    bad = np.zeros_like(truth)  # predicts one class always
    out = bootstrap.compare(truth, good, bad, macro_f1, resamples=400)
    assert out.significant and out.delta.low > 0
    assert out.prob_positive > 0.99


def test_a_tiny_gap_on_a_small_set_is_not_significant():
    # The outcome the report has to be able to state honestly.
    rng = np.random.default_rng(3)
    truth = rng.integers(0, 2, 200)
    a = np.where(rng.random(200) < 0.72, truth, 1 - truth)
    b = np.where(rng.random(200) < 0.70, truth, 1 - truth)
    assert not bootstrap.compare(truth, a, b, accuracy_score, resamples=600).significant


def test_results_are_reproducible():
    truth = np.array([0, 1] * 50)
    pred = np.roll(truth, 3)
    first = bootstrap.ci(truth, pred, accuracy_score, resamples=200, seed=7)
    second = bootstrap.ci(truth, pred, accuracy_score, resamples=200, seed=7)
    assert first == second
