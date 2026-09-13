import numpy as np

from brandbot.eval import metrics

LABELS = ["billing", "playback", "account"]


def test_perfect_predictions():
    truth = np.array(LABELS * 20)
    out = metrics.report(truth, truth, LABELS, resamples=200)
    assert out.macro_f1.point == 1.0
    assert out.accuracy.point == 1.0


def test_majority_guessing_has_low_macro_f1_despite_decent_accuracy():
    truth = np.array(["billing"] * 80 + ["playback"] * 10 + ["account"] * 10)
    pred = np.array(["billing"] * 100)
    out = metrics.report(truth, pred, LABELS, resamples=200)
    assert out.accuracy.point == 0.80
    assert out.macro_f1.point < 0.35


def test_prior_reweighting_differs_from_stratified_accuracy():
    # Stratified set: equal support. Real traffic: mostly billing.
    truth = np.array(["billing"] * 30 + ["playback"] * 30 + ["account"] * 30)
    pred = np.array(["billing"] * 30 + ["playback"] * 30 + ["billing"] * 30)  # account always wrong
    prior = {"billing": 0.7, "playback": 0.2, "account": 0.1}
    out = metrics.report(truth, pred, LABELS, prior=prior, resamples=200)
    assert out.accuracy.point == 2 / 3
    # Real traffic rarely sends the intent we fail, so deployment looks better.
    assert out.prior_weighted_accuracy > out.accuracy.point


def test_worst_classes_are_surfaced():
    truth = np.array(["billing"] * 20 + ["playback"] * 20 + ["account"] * 20)
    pred = np.array(["billing"] * 20 + ["playback"] * 20 + ["billing"] * 20)
    assert metrics.report(truth, pred, LABELS, resamples=100).worst(1)[0].label == "account"


def test_confusion_matrix_orientation():
    truth = np.array(["billing", "playback"])
    pred = np.array(["playback", "playback"])
    out = metrics.report(truth, pred, LABELS, resamples=50)
    i, j = LABELS.index("billing"), LABELS.index("playback")
    assert out.confusion[i][j] == 1  # rows are truth, columns are prediction


GROUPS = {"billing": "human", "playback": "troubleshoot", "account": "human"}


def test_same_step_forgives_a_confusion_inside_one_group():
    truth = np.array(["billing"] * 50)
    pred = np.array(["account"] * 50)  # wrong label, same handling group
    out = metrics.report(truth, pred, LABELS, groups=GROUPS, resamples=200)
    assert out.accuracy.point == 0.0
    assert out.same_step_accuracy.point == 1.0


def test_same_step_still_charges_for_crossing_a_group():
    truth = np.array(["billing"] * 50)
    pred = np.array(["playback"] * 50)
    out = metrics.report(truth, pred, LABELS, groups=GROUPS, resamples=200)
    assert out.same_step_accuracy.point == 0.0


def test_same_step_is_never_the_stricter_number():
    # It is the coarser measure, so quoting it alone would always flatter.
    truth = np.array(["billing", "playback", "account", "billing"])
    pred = np.array(["account", "playback", "account", "playback"])
    out = metrics.report(truth, pred, LABELS, groups=GROUPS, resamples=100)
    assert out.same_step_accuracy.point >= out.accuracy.point


def test_no_grouping_means_no_same_step_number():
    truth = np.array(LABELS * 10)
    assert metrics.report(truth, truth, LABELS, resamples=50).same_step_accuracy is None
