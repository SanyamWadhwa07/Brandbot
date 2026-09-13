"""Turn recorded runs and human labels into the numbers the report quotes.

Everything here is arithmetic over committed files. No model is called, no
embedding is computed, nothing touches the network. That is what lets a reviewer
reproduce the headline figures from a fresh clone with no keys.

Three numbers carry the argument, and they are deliberately phrased as support
operations rather than as scores:

  coverage   of 100 arriving messages, how many go out without a human
  risk       of the ones sent, how many a support lead would have stopped
  touches    human touches per 100 messages, counting escalations as one and a
             bad auto-reply as several, because apologising for a wrong answer
             costs more than answering in the first place

Coverage alone is meaningless: escalate everything and risk is zero. Risk alone is
meaningless for the same reason in reverse. They are only ever quoted together.

Intent accuracy is reported four ways because the four disagree and picking the
flattering one is the specific failure this project exists to avoid. Macro-F1
weights every intent equally, which the golden set is stratified to support.
Prior-weighted accuracy reweights to the mix real traffic actually has. Same-step
accuracy asks whether a mistake changed what happens to the customer. Plain
accuracy is the one to distrust.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brandbot.eval import bootstrap, metrics
from brandbot.eval.bootstrap import Comparison, Interval
from brandbot.eval.run import Run
from brandbot.gold import spec, store
from brandbot.gold.store import Labelled
from brandbot.intents import handling, taxonomy

# One escalation is one human touch. A reply that should not have gone out costs
# the apology, the re-handling and the damage, and five is the conventional
# placeholder rather than a measured figure for this brand. The report sweeps it.
COST_RATIO = 5.0


@dataclass(frozen=True)
class Scorecard:
    system: str
    n: int
    coverage: float
    risk: float
    touches_per_100: float
    macro_f1: Interval
    accuracy: Interval
    prior_weighted: float
    same_step: Interval
    route_accuracy: Interval
    route_accuracy_unforced: float
    n_unforced: int
    intent: metrics.Report


def _same_step(truth, predicted) -> float:
    return float(np.mean([handling.same_step(t, p) for t, p in zip(truth, predicted)]))


def _route_match(truth, predicted) -> float:
    return float(np.mean(np.asarray(truth) == np.asarray(predicted)))


def score(run: Run, labelled: list[Labelled], *, cost_ratio: float = COST_RATIO) -> Scorecard:
    handled = run.by_thread()
    verdicts = run.verdict_by_thread()
    rows = [r for r in labelled if r.thread_id in handled]

    truth_intent = [r.intent for r in rows]
    predicted_intent = [handled[r.thread_id].intent for r in rows]
    truth_route = [r.route for r in rows]
    predicted_route = [handled[r.thread_id].route for r in rows]

    auto = [r for r in rows if handled[r.thread_id].route == spec.AUTO]
    bad = [r for r in auto if r.thread_id in verdicts and not verdicts[r.thread_id].send]
    coverage = len(auto) / len(rows)
    risk = len(bad) / len(auto) if auto else 0.0

    unforced = [r for r in rows if not handled[r.thread_id].forced]

    return Scorecard(
        system=run.system,
        n=len(rows),
        coverage=coverage,
        risk=risk,
        touches_per_100=100 * ((len(rows) - len(auto)) + cost_ratio * len(bad)) / len(rows),
        macro_f1=bootstrap.ci(truth_intent, predicted_intent, metrics.macro_f1),
        accuracy=bootstrap.ci(truth_intent, predicted_intent, metrics.accuracy),
        prior_weighted=metrics.prior_weighted_accuracy(
            truth_intent, predicted_intent, store.prior(labelled)
        ),
        same_step=bootstrap.ci(truth_intent, predicted_intent, _same_step),
        route_accuracy=bootstrap.ci(truth_route, predicted_route, _route_match),
        route_accuracy_unforced=(
            _route_match(
                [r.route for r in unforced],
                [handled[r.thread_id].route for r in unforced],
            )
            if unforced
            else float("nan")
        ),
        n_unforced=len(unforced),
        intent=metrics.report(
            truth_intent, predicted_intent, taxonomy.names(), prior=store.prior(labelled)
        ),
    )


def against(a: Run, b: Run, labelled: list[Labelled]) -> dict[str, Comparison]:
    """Paired comparison of two systems on the same examples."""
    ha, hb = a.by_thread(), b.by_thread()
    rows = [r for r in labelled if r.thread_id in ha and r.thread_id in hb]
    truth = [r.intent for r in rows]
    pa = [ha[r.thread_id].intent for r in rows]
    pb = [hb[r.thread_id].intent for r in rows]
    return {
        "macro_f1": bootstrap.compare(truth, pa, pb, metrics.macro_f1),
        "same_step": bootstrap.compare(truth, pa, pb, _same_step),
    }
