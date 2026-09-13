"""The rating page must not tell the human what it is trying to measure.

A rater who can see which system wrote a reply, or what the judge already said
about it, is no longer an independent second opinion, and the agreement figure
computed from their answers would be measuring anchoring.
"""

from __future__ import annotations

import json

from brandbot.eval import rater
from brandbot.eval.judge import Verdict
from brandbot.eval.run import Run
from brandbot.gold import spec
from brandbot.pipeline.agent import Handling


def handling(thread_id: int, reply: str, route: str = spec.AUTO) -> Handling:
    return Handling(
        thread_id=thread_id, message=f"message {thread_id}", intent="playback_error",
        confidence=0.9, route=route, reason="", reply=reply, cited=[],
        top_score=0.8, margin=0.1, forced=False,
    )


def run(system: str, n: int) -> Run:
    handlings = [handling(i, f"{system} reply {i}") for i in range(n)]
    verdicts = [
        Verdict(i, True, True, True, True, i % 2 == 0, "because") for i in range(n)
    ]
    return Run(system, handlings, verdicts)


def test_sampling_is_deterministic():
    runs = [run("agent", 40)]
    assert [s.id for s in rater.draw(runs, per_system=5)] == [
        s.id for s in rater.draw(runs, per_system=5)
    ]


def test_escalated_messages_are_never_rated():
    handlings = [handling(0, "kept", spec.AUTO), handling(1, "", spec.ESCALATE)]
    drawn = rater.draw([Run("agent", handlings, [])], per_system=10)
    assert [s.thread_id for s in drawn] == [0]


def test_every_system_is_represented():
    runs = [run("agent", 40), run("nearest", 40), run("majority", 40)]
    drawn = rater.draw(runs, per_system=7)
    assert {s.system for s in drawn} == {"agent", "nearest", "majority"}
    assert len(drawn) == 21


def test_page_hides_system_and_verdict(tmp_path):
    runs = [run("agent", 6), run("majority", 6)]
    drawn = rater.draw(runs, per_system=3)
    out = rater.render(drawn, tmp_path / "rate.html", "judge-check")

    payload = json.loads(
        out.read_text(encoding="utf-8")
        .split('type="application/json">')[1]
        .split("</script>")[0]
        .replace("<\\/", "</")
    )
    for example in payload["examples"]:
        assert "system" not in example
        assert "send" not in example
        assert "verdict" not in example


def test_key_records_the_answer_separately(tmp_path):
    drawn = rater.draw([run("agent", 4)], per_system=2)
    path = rater.key(drawn, tmp_path / "key.jsonl")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {r["system"] for r in rows} == {"agent"}
