"""Run every system over the golden messages and record what each one did.

Split in two on purpose. This module makes the live calls and writes what came
back; `eval` reads those files and computes the numbers. A reviewer cloning the
repo runs only the second half, with no keys, no Ollama and no network, and still
recomputes every headline figure from the committed record.

Only replies that were actually going to be sent get judged. An escalated message
has no draft, and grading the empty string would quietly count a handover as a
reply failure.

The two baselines make no model calls at all. That is worth noticing rather than
optimising away: the whole cost of this system is the classifier and the drafter,
and the report has to show they buy something that retrieval alone does not.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from brandbot import config
from brandbot.baselines import majority, nearest
from brandbot.data import brand_slice, splits
from brandbot.eval import judge
from brandbot.eval.judge import Verdict
from brandbot.gold import spec
from brandbot.llm.budget import Budget
from brandbot.pipeline import agent
from brandbot.pipeline.agent import Handling
from brandbot.retrieval import index
from brandbot.retrieval.index import Index

RUNS = config.ARTIFACTS / "runs"

AGENT = "agent"
NEAREST = "nearest"
MAJORITY = "majority"


@dataclass(frozen=True)
class Run:
    system: str
    handlings: list[Handling]
    verdicts: list[Verdict]

    def by_thread(self) -> dict[int, Handling]:
        return {h.thread_id: h for h in self.handlings}

    def verdict_by_thread(self) -> dict[int, Verdict]:
        return {v.thread_id: v for v in self.verdicts}


def save(run: Run, root: Path = RUNS) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{run.system}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "system": run.system,
                "handlings": [asdict(h) for h in run.handlings],
                "verdicts": [asdict(v) for v in run.verdicts],
            },
            f,
            ensure_ascii=False,
            indent=1,
        )
    return path


def load(system: str, root: Path = RUNS) -> Run:
    with (root / f"{system}.json").open(encoding="utf-8") as f:
        raw = json.load(f)
    return Run(
        system=raw["system"],
        handlings=[Handling(**h) for h in raw["handlings"]],
        verdicts=[Verdict(**v) for v in raw["verdicts"]],
    )


def _judge_all(
    handlings: list[Handling], ix: Index, budget: Budget | None, replay_only: bool
) -> list[Verdict]:
    return [
        judge.grade(
            h.thread_id, h.message, h.reply, ix.search(h.message, agent.K),
            budget=budget, replay_only=replay_only,
        )
        for h in handlings
        if h.route == spec.AUTO and h.reply
    ]


def execute(
    system: str,
    messages: list[tuple[int, str]],
    *,
    budget: Budget | None = None,
    replay_only: bool = False,
) -> Run:
    parts = splits.partition(brand_slice.load())
    ix = index.build(parts[splits.Split.INDEX])

    if system == AGENT:
        handlings = [
            agent.handle(tid, text, ix, budget=budget, replay_only=replay_only)
            for tid, text in messages
        ]
    elif system == NEAREST:
        model = nearest.Nearest(ix, parts[splits.Split.DEV])
        handlings = [model.handle(tid, text) for tid, text in messages]
    elif system == MAJORITY:
        handlings = [majority.handle(tid, text) for tid, text in messages]
    else:
        raise ValueError(f"unknown system: {system}")

    return Run(system, handlings, _judge_all(handlings, ix, budget, replay_only))
