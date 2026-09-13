"""Confidence intervals and paired comparisons by resampling.

A point estimate on 200 examples is not a result. The question that decides the
headline is whether the agent's margin over the simple baseline survives the
noise in a set this size, and that needs an interval, not a difference of means.

Comparisons are paired: both systems are scored on the *same* resampled examples,
so the shared difficulty of those examples cancels instead of inflating the
spread. Comparing two independent intervals would be the wrong test and would
usually be the more flattering one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from brandbot import config

RESAMPLES = 2_000
ALPHA = 0.05

Metric = Callable[[Sequence, Sequence], float]


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


@dataclass(frozen=True)
class Comparison:
    delta: Interval
    prob_positive: float

    @property
    def significant(self) -> bool:
        """True when the interval for the difference excludes zero."""
        return self.delta.low > 0 or self.delta.high < 0


def _indices(n: int, resamples: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(resamples, n))


def ci(
    truth: Sequence,
    predicted: Sequence,
    metric: Metric,
    *,
    resamples: int = RESAMPLES,
    alpha: float = ALPHA,
    seed: int = config.SEED,
) -> Interval:
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    draws = [
        metric(truth[idx], predicted[idx]) for idx in _indices(len(truth), resamples, seed)
    ]
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return Interval(float(metric(truth, predicted)), float(low), float(high))


def compare(
    truth: Sequence,
    a: Sequence,
    b: Sequence,
    metric: Metric,
    *,
    resamples: int = RESAMPLES,
    alpha: float = ALPHA,
    seed: int = config.SEED,
) -> Comparison:
    """Distribution of metric(a) - metric(b) under resampling of the examples."""
    truth, a, b = np.asarray(truth), np.asarray(a), np.asarray(b)
    deltas = np.array(
        [
            metric(truth[idx], a[idx]) - metric(truth[idx], b[idx])
            for idx in _indices(len(truth), resamples, seed)
        ]
    )
    low, high = np.quantile(deltas, [alpha / 2, 1 - alpha / 2])
    point = metric(truth, a) - metric(truth, b)
    return Comparison(Interval(float(point), float(low), float(high)), float((deltas > 0).mean()))
