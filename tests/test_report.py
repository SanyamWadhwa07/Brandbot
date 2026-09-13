"""The scorecard has to behave correctly at the edges that matter operationally.

A system that escalates everything and a system that sends everything are the two
poles the escalation work sits between, and both are easy to score wrongly: the
first has no sent replies to divide by, the second has no escalations. Getting
either wrong would make the headline coverage and risk figures meaningless in
exactly the region where they decide something.
"""

from __future__ import annotations

from brandbot.eval import report
from brandbot.eval.judge import Verdict
from brandbot.eval.run import Run
from brandbot.gold import spec
from brandbot.gold.sample import ENRICHED, RANDOM
from brandbot.gold.store import Labelled
from brandbot.pipeline.agent import Handling

INTENTS = ["playback_error", "billing_refund", "content_availability", "app_bug"]


def labelled(n: int = 12) -> list[Labelled]:
    return [
        Labelled(
            thread_id=i,
            text=f"message {i}",
            intent=INTENTS[i % len(INTENTS)],
            route=spec.AUTO if i % 3 else spec.ESCALATE,
            multi_intent=False,
            revealed_reply=False,
            stratum=RANDOM if i < 8 else ENRICHED,
        )
        for i in range(n)
    ]


def run(system: str, rows: list[Labelled], *, route: str, correct: bool, send: bool) -> Run:
    handlings = [
        Handling(
            thread_id=r.thread_id, message=r.text,
            intent=r.intent if correct else "other",
            confidence=0.9, route=route, reason="", reply="a reply" if route == spec.AUTO else "",
            cited=[], top_score=0.8, margin=0.1, forced=False,
        )
        for r in rows
    ]
    verdicts = [
        Verdict(h.thread_id, True, True, True, True, send, "") for h in handlings
        if h.route == spec.AUTO
    ]
    return Run(system, handlings, verdicts)


def test_perfect_classifier_scores_one():
    rows = labelled()
    card = report.score(run("agent", rows, route=spec.AUTO, correct=True, send=True), rows)
    assert card.macro_f1.point == 1.0
    assert card.same_step.point == 1.0


def test_sending_everything_well_is_full_coverage_and_no_risk():
    rows = labelled()
    card = report.score(run("a", rows, route=spec.AUTO, correct=True, send=True), rows)
    assert card.coverage == 1.0
    assert card.risk == 0.0
    assert card.touches_per_100 == 0.0


def test_sending_everything_badly_costs_the_full_penalty():
    rows = labelled()
    card = report.score(
        run("a", rows, route=spec.AUTO, correct=True, send=False), rows, cost_ratio=5.0
    )
    assert card.coverage == 1.0
    assert card.risk == 1.0
    assert card.touches_per_100 == 500.0


def test_escalating_everything_is_zero_coverage_and_zero_risk():
    rows = labelled()
    card = report.score(run("a", rows, route=spec.ESCALATE, correct=True, send=True), rows)
    assert card.coverage == 0.0
    # Risk is undefined with nothing sent; it must read as zero rather than divide.
    assert card.risk == 0.0
    assert card.touches_per_100 == 100.0


def test_harmless_confusion_scores_better_on_same_step_than_on_f1():
    rows = [
        Labelled(i, f"m{i}", "app_bug", spec.AUTO, False, False, RANDOM) for i in range(10)
    ]
    handlings = [
        Handling(r.thread_id, r.text, "service_outage", 0.9, spec.AUTO, "", "x", [], 0.8, 0.1, False)
        for r in rows
    ]
    card = report.score(Run("a", handlings, []), rows)
    assert card.macro_f1.point == 0.0
    assert card.same_step.point == 1.0


def test_paired_comparison_favours_the_better_system():
    rows = labelled()
    good = run("agent", rows, route=spec.AUTO, correct=True, send=True)
    bad = run("nearest", rows, route=spec.AUTO, correct=False, send=True)
    assert report.against(good, bad, rows)["macro_f1"].delta.point > 0
