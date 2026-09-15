"""Judge-human agreement is computed by joining committed files, not by re-rating."""

from __future__ import annotations

import json

from brandbot.eval import judge_check
from brandbot.eval.judge import Verdict
from brandbot.eval.run import Run, save
from brandbot.pipeline.agent import Handling
from brandbot.gold import spec


def handling(thread_id: int) -> Handling:
    return Handling(
        thread_id=thread_id, message=f"message {thread_id}", intent="playback_error",
        confidence=0.9, route=spec.AUTO, reason="", reply=f"reply {thread_id}", cited=[],
        top_score=0.8, margin=0.1, forced=False,
    )


def write_run(root, system: str, verdicts: dict[int, bool]) -> None:
    handlings = [handling(tid) for tid in verdicts]
    vs = [Verdict(tid, send, send, send, send, send, "") for tid, send in verdicts.items()]
    save(Run(system, handlings, vs), root=root)


def test_matches_human_to_judge_by_system_and_thread(tmp_path):
    runs_root = tmp_path / "runs"
    write_run(runs_root, "agent", {1: True, 2: False})
    write_run(runs_root, "nearest", {3: True})

    key = tmp_path / "key.jsonl"
    key.write_text("\n".join(json.dumps(r) for r in [
        {"id": "agent-1", "system": "agent", "thread_id": 1},
        {"id": "agent-2", "system": "agent", "thread_id": 2},
        {"id": "nearest-3", "system": "nearest", "thread_id": 3},
    ]), encoding="utf-8")

    ratings = tmp_path / "ratings.jsonl"
    ratings.write_text("\n".join(json.dumps(r) for r in [
        {"id": "agent-1", "thread_id": 1, "send": True, "failed": [], "peeked": False},
        {"id": "agent-2", "thread_id": 2, "send": True, "failed": [], "peeked": False},
        {"id": "nearest-3", "thread_id": 3, "send": True, "failed": [], "peeked": True},
    ]), encoding="utf-8")

    matched = judge_check.match(ratings, key, runs_root)
    assert len(matched) == 3
    assert {(m.system, m.thread_id, m.human_send, m.judge_send) for m in matched} == {
        ("agent", 1, True, True), ("agent", 2, True, False), ("nearest", 3, True, True),
    }


def test_unmatched_ids_are_skipped(tmp_path):
    runs_root = tmp_path / "runs"
    write_run(runs_root, "agent", {1: True})

    key = tmp_path / "key.jsonl"
    key.write_text(json.dumps({"id": "agent-1", "system": "agent", "thread_id": 1}), encoding="utf-8")

    ratings = tmp_path / "ratings.jsonl"
    ratings.write_text("\n".join(json.dumps(r) for r in [
        {"id": "agent-1", "thread_id": 1, "send": True, "failed": [], "peeked": False},
        {"id": "agent-99", "thread_id": 99, "send": False, "failed": [], "peeked": False},
    ]), encoding="utf-8")

    matched = judge_check.match(ratings, key, runs_root)
    assert len(matched) == 1


def test_score_reports_overall_and_per_system(tmp_path):
    runs_root = tmp_path / "runs"
    write_run(runs_root, "agent", {i: (i % 2 == 0) for i in range(20)})

    key = tmp_path / "key.jsonl"
    key.write_text("\n".join(
        json.dumps({"id": f"agent-{i}", "system": "agent", "thread_id": i}) for i in range(20)
    ), encoding="utf-8")

    ratings = tmp_path / "ratings.jsonl"
    ratings.write_text("\n".join(
        json.dumps({"id": f"agent-{i}", "thread_id": i, "send": (i % 2 == 0), "failed": [], "peeked": False})
        for i in range(20)
    ), encoding="utf-8")

    matched = judge_check.match(ratings, key, runs_root)
    scored = judge_check.score(matched)
    assert "overall" in scored and "agent" in scored
    assert scored["overall"].kappa.point == 1.0
