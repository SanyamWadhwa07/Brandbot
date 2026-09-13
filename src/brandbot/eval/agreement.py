"""Agreement between two raters, chance-corrected.

Raw percent agreement is the number everyone quotes and the one that misleads.
On a label that is 85% "pass", two raters who both guess "pass" every time agree
85% of the time while carrying no information at all. Cohen's kappa subtracts the
agreement expected by chance, so it is the number that belongs in the report.

Both are returned together, deliberately: the gap between them is itself
evidence about how skewed the label is.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import cohen_kappa_score

from brandbot import config
from brandbot.eval.bootstrap import Interval

# Landis and Koch (1977). Conventional, arbitrary, and worth stating rather than
# leaving the reader to guess what counts as good.
BANDS = ((0.81, "almost perfect"), (0.61, "substantial"), (0.41, "moderate"),
         (0.21, "fair"), (0.0, "slight"), (-1.0, "worse than chance"))


@dataclass(frozen=True)
class Agreement:
    kappa: Interval
    raw: float
    n: int

    @property
    def interpretation(self) -> str:
        return next(label for floor, label in BANDS if self.kappa.point >= floor)

    def __str__(self) -> str:
        return (
            f"kappa {self.kappa} ({self.interpretation}), "
            f"raw agreement {self.raw:.1%}, n={self.n}"
        )


def measure(
    a: Sequence,
    b: Sequence,
    *,
    resamples: int = 2_000,
    alpha: float = 0.05,
    seed: int = config.SEED,
) -> Agreement:
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(seed)

    draws = []
    for idx in rng.integers(0, len(a), size=(resamples, len(a))):
        # A resample can contain a single label, where kappa is undefined.
        if len(np.unique(a[idx])) < 2 or len(np.unique(b[idx])) < 2:
            continue
        draws.append(cohen_kappa_score(a[idx], b[idx]))

    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2]) if draws else (np.nan, np.nan)
    return Agreement(
        kappa=Interval(float(cohen_kappa_score(a, b)), float(low), float(high)),
        raw=float((a == b).mean()),
        n=len(a),
    )
