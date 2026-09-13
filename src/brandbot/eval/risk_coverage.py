"""Escalation as selective prediction.

The agent does not have to answer everything. It has to answer what it answers
well, and hand the rest to a human. That makes the useful question not "how
often is escalation correct" but "how bad are the replies we actually sent, as a
function of how many we sent".

Risk here is the rubric failure rate among auto-sent replies. Coverage is the
share auto-sent. A system can always reach zero risk by escalating everything,
which is why neither number means anything alone.

Content rules are applied first and are not negotiable: a billing dispute goes to
a human however confident the model is about its draft.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Point:
    threshold: float
    coverage: float
    risk: float
    expected_cost: float


@dataclass(frozen=True)
class Curve:
    points: list[Point]
    signal: str

    @property
    def aurc(self) -> float:
        """Area under the risk-coverage curve. Lower is a better signal.

        Summarises the whole curve in one number so several confidence signals can
        be ranked without picking an operating point first.
        """
        pts = sorted(self.points, key=lambda p: p.coverage)
        cov = np.array([p.coverage for p in pts])
        risk = np.array([p.risk for p in pts])
        return float(np.trapezoid(risk, cov)) if len(pts) > 1 else float("nan")

    def at_cost_ratio(self, ratio: float) -> Point:
        """Cheapest operating point when a bad auto-reply costs `ratio` human touches."""
        return min(self.points, key=lambda p: p.expected_cost if p.threshold else float("inf"))


def curve(
    confidence: Sequence[float],
    failed: Sequence[bool],
    *,
    must_escalate: Sequence[bool] | None = None,
    cost_ratio: float = 5.0,
    signal: str = "confidence",
) -> Curve:
    confidence = np.asarray(confidence, dtype=float)
    failed = np.asarray(failed, dtype=bool)
    forced = (
        np.zeros(len(confidence), dtype=bool)
        if must_escalate is None
        else np.asarray(must_escalate, dtype=bool)
    )

    points = []
    for threshold in np.unique(np.r_[confidence, confidence.max() + 1e-9]):
        auto = (confidence >= threshold) & ~forced
        n_auto = int(auto.sum())
        n_bad = int((auto & failed).sum())
        points.append(
            Point(
                threshold=float(threshold),
                coverage=n_auto / len(confidence),
                risk=(n_bad / n_auto) if n_auto else 0.0,
                # One unit is a human touch; a bad auto-reply costs `ratio` of them.
                expected_cost=(n_bad * cost_ratio + (len(confidence) - n_auto)) / len(confidence),
            )
        )
    return Curve(points, signal)
