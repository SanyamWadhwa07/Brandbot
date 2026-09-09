"""Reconstruct conversations from the reply graph.

`in_response_to_tweet_id` gives each tweet its parent, so a conversation is a
connected component of that forest. Roots are resolved by pointer doubling
rather than recursion: conversations are shallow, and 2.8M Python-level walks
would dominate the runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brandbot.data.ingest import ReplyGraph


@dataclass(frozen=True)
class Threads:
    """One entry per conversation, aligned across all arrays."""

    root_row: np.ndarray  # int64, row index of the opening tweet
    brand: np.ndarray  # int32 author code of the replying brand, -1 if never answered
    size: np.ndarray  # int32 tweets in the conversation
    started_at: np.ndarray  # int64 epoch seconds of the opening tweet
    customer_opened: np.ndarray  # bool
    n_customers: np.ndarray  # int32 distinct inbound authors

    def __len__(self) -> int:
        return len(self.root_row)

    def is_conversation(self) -> np.ndarray:
        """Threads that are one customer talking to one brand.

        Reply chains fan in: when a brand posts a broadcast, hundreds of unrelated
        customers reply beneath it and pointer-doubling merges them into a single
        component. The largest such component here holds 972 distinct customers.
        Those are not conversations and would poison both retrieval and the golden
        set, so anything with more than one customer is dropped.
        """
        return (self.n_customers == 1) & self.customer_opened


def resolve_roots(graph: ReplyGraph) -> np.ndarray:
    """Row index of each tweet's conversation root."""
    order = np.argsort(graph.tweet_id, kind="stable")
    sorted_ids = graph.tweet_id[order]

    pos = np.searchsorted(sorted_ids, graph.parent)
    pos = np.clip(pos, 0, len(sorted_ids) - 1)
    # A parent can point outside the corpus; those tweets are treated as roots.
    resolved = sorted_ids[pos] == graph.parent
    parent_row = np.where(resolved, order[pos], np.arange(len(graph)))

    # Self-loops at roots make pointer doubling converge without special-casing.
    anc = parent_row
    while True:
        nxt = anc[anc]
        if np.array_equal(nxt, anc):
            return anc
        anc = nxt


def build_threads(graph: ReplyGraph, roots: np.ndarray) -> Threads:
    root_rows, inverse, sizes = np.unique(roots, return_inverse=True, return_counts=True)

    brand = np.full(len(root_rows), -1, dtype=np.int32)
    # Later rows overwrite earlier ones, which is harmless: a conversation is with
    # a single brand, so every outbound tweet in it carries the same author.
    outbound = ~graph.inbound
    brand[inverse[outbound]] = graph.author[outbound]

    inb = graph.inbound
    seen = np.unique(np.stack([inverse[inb], graph.author[inb]]), axis=1)
    n_customers = np.bincount(seen[0], minlength=len(root_rows)).astype(np.int32)

    return Threads(
        root_row=root_rows,
        brand=brand,
        size=sizes.astype(np.int32),
        started_at=graph.created_at[root_rows],
        customer_opened=graph.inbound[root_rows],
        n_customers=n_customers,
    )


def turns_by_thread(roots: np.ndarray, graph: ReplyGraph) -> dict[int, np.ndarray]:
    """Row indices of each conversation's tweets, in chronological order."""
    order = np.lexsort((graph.created_at, roots))
    grouped = roots[order]
    boundaries = np.flatnonzero(np.diff(grouped)) + 1
    return {
        int(grouped[start]): order[start:stop]
        for start, stop in zip(
            np.r_[0, boundaries], np.r_[boundaries, len(order)], strict=True
        )
    }
