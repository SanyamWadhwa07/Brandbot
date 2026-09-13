import numpy as np

from brandbot.eval import agreement


def test_perfect_agreement():
    labels = np.array([0, 1, 0, 1, 1, 0] * 20)
    out = agreement.measure(labels, labels, resamples=200)
    assert out.kappa.point == 1.0
    assert out.raw == 1.0
    assert out.interpretation == "almost perfect"


def test_skewed_labels_expose_the_gap_raw_agreement_hides():
    # Both raters say "pass" almost always. Raw agreement looks strong; kappa does not.
    rng = np.random.default_rng(0)
    a = (rng.random(400) < 0.9).astype(int)
    b = (rng.random(400) < 0.9).astype(int)
    out = agreement.measure(a, b, resamples=400)
    assert out.raw > 0.80
    assert out.kappa.point < 0.15


def test_disagreement_scores_near_zero():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2, 300)
    b = rng.integers(0, 2, 300)
    out = agreement.measure(a, b, resamples=300)
    assert abs(out.kappa.point) < 0.2


def test_interpretation_bands():
    labels = np.array([0, 1] * 50)
    assert agreement.measure(labels, labels, resamples=100).interpretation == "almost perfect"


def test_reproducible():
    rng = np.random.default_rng(2)
    a, b = rng.integers(0, 2, 120), rng.integers(0, 2, 120)
    assert agreement.measure(a, b, resamples=200, seed=5) == agreement.measure(a, b, resamples=200, seed=5)
