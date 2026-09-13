"""Load the golden set and join it to the record of how it was sampled.

The labelled file holds only what a human wrote. Which stratum an example came
from lives in the sampling record instead, because that field was derived from an
embedding model and `data/gold/` is the one place nothing model-derived is allowed
to reach. Joining at load time keeps both true at once.

`random` examples are the only unbiased view of what real traffic looks like, so
they alone estimate the intent mix the prior-weighted numbers reweight to. The
`enriched` examples exist so rare intents have enough examples to score at all,
and counting them into the prior would overstate how often rare problems arrive.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from brandbot import config
from brandbot.gold.sample import CANDIDATES_PATH, ENRICHED, RANDOM

GOLDEN_PATH = config.GOLD / "golden.jsonl"


@dataclass(frozen=True)
class Labelled:
    thread_id: int
    text: str
    intent: str
    route: str
    multi_intent: bool
    revealed_reply: bool
    stratum: str


def load(
    path: Path = GOLDEN_PATH, candidates: Path = CANDIDATES_PATH
) -> list[Labelled]:
    strata = {
        int(r["thread_id"]): r["stratum"]
        for r in map(json.loads, candidates.read_text(encoding="utf-8").splitlines())
    }
    out = []
    for row in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
        tid = int(row["thread_id"])
        out.append(
            Labelled(
                thread_id=tid,
                text=row["text"],
                intent=row["intent"],
                route=row["route"],
                multi_intent=bool(row.get("multi_intent", False)),
                revealed_reply=bool(row.get("revealed_reply", False)),
                stratum=strata[tid],
            )
        )
    return out


def prior(labelled: list[Labelled]) -> dict[str, float]:
    """Intent mix of real traffic, from the uniformly drawn examples only."""
    drawn = [row.intent for row in labelled if row.stratum == RANDOM]
    return {intent: drawn.count(intent) / len(drawn) for intent in set(drawn)}


def counts(labelled: list[Labelled]) -> dict[str, int]:
    return {
        RANDOM: sum(1 for r in labelled if r.stratum == RANDOM),
        ENRICHED: sum(1 for r in labelled if r.stratum == ENRICHED),
        "revealed": sum(1 for r in labelled if r.revealed_reply),
        "multi_intent": sum(1 for r in labelled if r.multi_intent),
    }
