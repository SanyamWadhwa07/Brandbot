"""Classification metrics for the intent task.

Two accuracy numbers are reported and they differ. The golden set is stratified
across intents so that rare ones are measurable at all, which means its intent mix
is not the mix real traffic has. Macro-F1 over that set answers "how well does
this do per intent"; it does not answer "what fraction of tomorrow's messages
would be classified correctly".

The second number reweights per-intent recall by the intent frequency actually
observed in unlabelled traffic. Reporting only the flattering one of the two is
the specific failure this project is meant to avoid.

A third number asks a different question. Macro-F1 charges the same price for
every confusion, but the taxonomy labels the topic of a complaint and several
topics lead to the same next step: on the dev split the action Hulu took after an
`app_bug` and after a `service_outage` is 96% the same. Same-step accuracy scores
a prediction correct when it lands in the same handling group, so the errors it
counts are the ones that change what happens to the customer. It is coarser and
will always look better than accuracy, which is why `report` returns both.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from brandbot.eval.bootstrap import Interval, ci


@dataclass(frozen=True)
class PerClass:
    label: str
    support: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class Report:
    macro_f1: Interval
    accuracy: Interval
    same_step_accuracy: Interval | None
    prior_weighted_accuracy: float | None
    per_class: list[PerClass]
    labels: list[str]
    confusion: np.ndarray

    def worst(self, n: int = 3) -> list[PerClass]:
        return sorted(self.per_class, key=lambda c: c.f1)[:n]


def macro_f1(truth: Sequence, predicted: Sequence) -> float:
    return float(f1_score(truth, predicted, average="macro", zero_division=0))


def accuracy(truth: Sequence, predicted: Sequence) -> float:
    return float(np.mean(np.asarray(truth) == np.asarray(predicted)))


def same_step_accuracy(
    truth: Sequence, predicted: Sequence, groups: dict[str, str]
) -> float:
    """Share of predictions that lead to the same next step, exact label or not.

    `groups` maps each label to the handling group it belongs to. Unknown labels
    raise rather than forming a group of their own, because a label outside the
    taxonomy means something upstream is broken and swallowing it would hide that.
    """
    pairs = zip(np.asarray(truth).tolist(), np.asarray(predicted).tolist(), strict=True)
    return float(np.mean([groups[t] == groups[p] for t, p in pairs]))


def prior_weighted_accuracy(
    truth: Sequence, predicted: Sequence, prior: dict[str, float]
) -> float:
    """Per-intent recall reweighted to the intent mix of real traffic.

    What the agent would score on an unstratified stream, estimated from a
    stratified evaluation set.
    """
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    total = 0.0
    for label, weight in prior.items():
        seen = truth == label
        if seen.any():
            total += weight * float((predicted[seen] == label).mean())
    return total / sum(prior.values())


def report(
    truth: Sequence,
    predicted: Sequence,
    labels: Sequence[str],
    *,
    prior: dict[str, float] | None = None,
    groups: dict[str, str] | None = None,
    resamples: int = 2_000,
) -> Report:
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    labels = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )
    return Report(
        macro_f1=ci(truth, predicted, macro_f1, resamples=resamples),
        accuracy=ci(truth, predicted, accuracy, resamples=resamples),
        same_step_accuracy=(
            ci(
                truth,
                predicted,
                lambda t, p: same_step_accuracy(t, p, groups),
                resamples=resamples,
            )
            if groups
            else None
        ),
        prior_weighted_accuracy=(
            prior_weighted_accuracy(truth, predicted, prior) if prior else None
        ),
        per_class=[
            PerClass(label, int(s), float(p), float(r), float(fs))
            for label, p, r, fs, s in zip(labels, precision, recall, f1, support, strict=True)
        ],
        labels=labels,
        confusion=confusion_matrix(truth, predicted, labels=labels),
    )
