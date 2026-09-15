"""Check the judge's `send` verdicts against a human's, on the same replies.

`brandbot rate` draws a blind sample and writes an answer key that says which
system wrote each reply, in `data/interim/rating_key.jsonl`. The key is never
shown to the human; it only lets this module join the human's rating back to the
judge verdict recorded in the same system's run. Both files are committed, so
this is pure arithmetic over records already on disk, same as `report.py`.

`peeked` is not a leakage flag. It only records whether the rater chose to reveal
the retrieved precedents while deciding, which is part of the rubric, not the
system identity or the judge's own answer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from brandbot import config
from brandbot.eval.agreement import Agreement, measure
from brandbot.eval.run import RUNS, load as load_run

RATINGS_PATH = config.GOLD / "ratings.jsonl"
KEY_PATH = config.DATA / "interim" / "rating_key.jsonl"


@dataclass(frozen=True)
class Matched:
    id: str
    system: str
    thread_id: int
    human_send: bool
    judge_send: bool


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def match(
    ratings_path: Path = RATINGS_PATH, key_path: Path = KEY_PATH, runs_root: Path = RUNS
) -> list[Matched]:
    """Join each human rating to the judge verdict for the same (system, reply)."""
    key = {row["id"]: row for row in _read_jsonl(key_path)}
    systems = {row["system"] for row in key.values()}
    verdicts = {s: load_run(s, root=runs_root).verdict_by_thread() for s in systems}

    out = []
    for row in _read_jsonl(ratings_path):
        entry = key.get(row["id"])
        if entry is None:
            continue
        v = verdicts.get(entry["system"], {}).get(entry["thread_id"])
        if v is None:
            continue
        out.append(Matched(
            id=row["id"], system=entry["system"], thread_id=entry["thread_id"],
            human_send=row["send"], judge_send=v.send,
        ))
    return out


def score(matched: list[Matched]) -> dict[str, Agreement]:
    """Agreement overall and per system. A system with too few matches is skipped."""
    out = {}
    if matched:
        out["overall"] = measure(
            [m.human_send for m in matched], [m.judge_send for m in matched]
        )
    by_system: dict[str, list[Matched]] = {}
    for m in matched:
        by_system.setdefault(m.system, []).append(m)
    for system, rows in sorted(by_system.items()):
        if len(rows) < 2:
            continue
        out[system] = measure([m.human_send for m in rows], [m.judge_send for m in rows])
    return out
