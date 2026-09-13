"""Retrieval with no model on top: find the closest past message, send what was sent then.

The one to beat, and it is not a straw man. Every reply it sends is real text this
brand actually published, so it cannot hallucinate a policy, invent a refund or
promise a release date. Whatever the agent's grounding score turns out to be, this
baseline's is the ceiling that score should be read against.

Its failure mode is specific and worth naming: it answers the question it was
trained on rather than the one it was asked. A message about Rick and Morty season
three retrieves a real reply about Rick and Morty season three, dated eighteen
months earlier, and sends it as though nothing has changed.

Intent comes from the same frozen dev clustering the golden set was sampled
through, so it needs no labels and no model call. That makes it the honest test of
whether the classifier earns its tokens.

Routing is one number: escalate when the nearest precedent is too far away. It gets
none of the agent's content rules, which is deliberate. Comparing it with the agent
measures the rules and the model together, and the report separates them by running
the agent again with the gate switched off.
"""

from __future__ import annotations

import numpy as np

from brandbot.data.brand_slice import Conversation
from brandbot.gold import spec
from brandbot.intents import induce, merge
from brandbot.pipeline.agent import Handling
from brandbot.retrieval import embed
from brandbot.retrieval.index import Index

FLOOR = 0.62


class Nearest:
    def __init__(self, ix: Index, dev: list[Conversation]) -> None:
        self.ix = ix
        vectors = embed.embed([c.opening["text"] for c in dev], embed.CLUSTERING)
        labels = induce.cluster(vectors)
        centroids = np.stack(
            [vectors[labels == c].mean(axis=0) for c in range(induce.N_CLUSTERS)]
        )
        self.centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
        self.intents = np.array([merge.CLUSTER_MERGE[c] for c in range(induce.N_CLUSTERS)])

    def handle(self, thread_id: int, message: str) -> Handling:
        hits = self.ix.search(message, 2)
        top = hits[0].score
        margin = top - hits[1].score if len(hits) > 1 else top

        query = embed.embed([message], embed.CLUSTERING)[0]
        intent = str(self.intents[int(np.argmax(self.centroids @ query))])

        if top < FLOOR:
            return Handling(
                thread_id, message, intent, top, spec.ESCALATE,
                f"nearest precedent scores {top:.2f}, nothing close enough to copy",
                "", [], top, margin, forced=False,
            )
        return Handling(
            thread_id, message, intent, top, spec.AUTO, "",
            hits[0].precedent.reply, [hits[0].precedent.thread_id], top, margin, forced=False,
        )
