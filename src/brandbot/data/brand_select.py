"""Pick the brand to build for, against a criterion fixed before it is run.

Choosing a brand after seeing which one scores best would make every downstream
number a selection artefact. The weights and thresholds below are committed
first; whichever brand wins, wins.

The criterion follows from what the agent has to do. Drafting a grounded reply
requires precedents that contain an actual resolution, so a brand whose replies
are mostly "please DM us" is unusable no matter how much volume it has.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from brandbot import config
from brandbot.data import ingest, threads
from brandbot.data.ingest import ReplyGraph
from brandbot.data.threads import Threads

MIN_CONVERSATIONS = 10_000
SAMPLE_THREADS = 3_000

# Replies that move the customer off Twitter without resolving anything. These
# carry no resolution content, so they are dead weight as retrieval precedents.
DM_REDIRECT = re.compile(
    r"\b(dm|pm)\b|direct message|private message|send us a message|message us"
    r"|shoot us a|click ['‘’\"]?message|via (dm|pm)\b",
    re.IGNORECASE,
)

# Weak proxy for a resolved conversation: the customer signs off positively.
THANKS = re.compile(r"\b(thanks|thank you|thankyou|thx|ty|appreciate(d)?)\b", re.IGNORECASE)


@dataclass(frozen=True)
class BrandScore:
    name: str
    conversations: int
    dm_redirect_rate: float
    median_reply_chars: float
    thanks_rate: float
    composite: float


def candidate_brands(graph: ReplyGraph, table: Threads) -> list[int]:
    keep = table.is_conversation()
    counts = np.bincount(table.brand[keep], minlength=len(graph.authors))
    return [int(i) for i in np.flatnonzero(counts >= MIN_CONVERSATIONS)]


def _sample_rows(
    graph: ReplyGraph, table: Threads, roots: np.ndarray, brand_code: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Rows of brand replies, and rows of each sampled conversation's final tweet."""
    keep = table.is_conversation() & (table.brand == brand_code)
    chosen = table.root_row[keep]
    if len(chosen) > SAMPLE_THREADS:
        chosen = rng.choice(chosen, SAMPLE_THREADS, replace=False)

    members = np.flatnonzero(np.isin(roots, chosen))
    replies = members[~graph.inbound[members]]

    order = np.lexsort((graph.created_at[members], roots[members]))
    ordered = members[order]
    last_of_thread = ordered[np.r_[np.flatnonzero(np.diff(roots[ordered])), len(ordered) - 1]]
    return replies, last_of_thread


def score_brands(graph: ReplyGraph, table: Threads, roots: np.ndarray) -> list[BrandScore]:
    rng = np.random.default_rng(config.SEED)
    codes = candidate_brands(graph, table)
    keep = table.is_conversation()
    volume = np.bincount(table.brand[keep], minlength=len(graph.authors))

    per_brand = {c: _sample_rows(graph, table, roots, c, rng) for c in codes}
    all_rows = np.unique(np.concatenate([np.r_[a, b] for a, b in per_brand.values()]))
    text = ingest.load_text(all_rows)

    raw = []
    for code in codes:
        replies, finals = per_brand[code]
        reply_text = [text[int(r)] for r in replies]
        final_text = [text[int(r)] for r in finals]
        raw.append(
            (
                graph.authors[code],
                int(volume[code]),
                float(np.mean([bool(DM_REDIRECT.search(t)) for t in reply_text])),
                float(np.median([len(t) for t in reply_text])),
                float(np.mean([bool(THANKS.search(t)) for t in final_text])),
            )
        )

    def z(values: list[float]) -> np.ndarray:
        arr = np.array(values)
        return (arr - arr.mean()) / (arr.std() or 1.0)

    # Equal weights. Redirect rate is inverted because low is good; the three
    # signals are otherwise on incomparable scales, so each is standardised.
    composite = -z([r[2] for r in raw]) + z([r[3] for r in raw]) + z([r[4] for r in raw])

    scores = [BrandScore(*r, composite=float(c)) for r, c in zip(raw, composite, strict=True)]
    return sorted(scores, key=lambda s: s.composite, reverse=True)


def load_all() -> tuple[ReplyGraph, Threads, np.ndarray]:
    graph = ingest.load()
    roots = threads.resolve_roots(graph)
    return graph, threads.build_threads(graph, roots), roots
