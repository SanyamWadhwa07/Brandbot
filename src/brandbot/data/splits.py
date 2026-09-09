"""Split conversations by time, not at random.

Retrieval always looks backwards in deployment: you answer today's message with
last month's precedents. A random split lets the agent retrieve a precedent
written after the message it is answering, which inflates every number and could
never hold in production.

Quantile cuts rather than fixed dates, so the same code splits a second brand
for the transfer probe.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from brandbot.data.brand_slice import Conversation

INDEX_END = 0.70
DEV_END = 0.88


class Split(StrEnum):
    INDEX = "index"  # retrieval corpus, earliest
    DEV = "dev"  # taxonomy induction and prompt iteration
    GOLD = "gold"  # golden set, latest


def assign(conversations: list[Conversation]) -> list[Split]:
    starts = np.array([c.started_at for c in conversations])
    order = np.argsort(starts, kind="stable")
    rank = np.empty(len(starts), dtype=np.int64)
    rank[order] = np.arange(len(starts))

    frac = rank / len(starts)
    return [
        Split.INDEX if f < INDEX_END else Split.DEV if f < DEV_END else Split.GOLD
        for f in frac
    ]


def partition(
    conversations: list[Conversation],
) -> dict[Split, list[Conversation]]:
    out: dict[Split, list[Conversation]] = {s: [] for s in Split}
    for conv, split in zip(conversations, assign(conversations), strict=True):
        out[split].append(conv)
    return out
