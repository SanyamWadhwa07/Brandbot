"""Discover candidate intents from the brand's own messages.

Deliberately over-clusters, then names each cluster, then leaves the merge to a
human. Picking "the right k" up front is guesswork; splitting too finely and
merging by eye is not, and the merge is where the domain judgement actually
belongs.

Runs on the dev split only. The gold split must never be looked at while the
taxonomy is being shaped, or the labels stop being independent of the label set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans

from brandbot import config
from brandbot.data.brand_slice import Conversation
from brandbot.llm import client
from brandbot.retrieval import embed

N_CLUSTERS = 20
SAMPLES_PER_CLUSTER = 10

PROPOSE_SYSTEM = """You are naming customer-support intent categories for a video
streaming service, from real customer messages that were grouped by similarity.

Give the group a short snake_case name and a one-line definition that says what
belongs in it and what does not. Name what the customer WANTS, not how they feel.
If the messages have no single theme, say so in the definition."""

PROPOSE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "definition": {"type": "string"},
        "coherent": {"type": "boolean"},
    },
    "required": ["name", "definition", "coherent"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Cluster:
    cluster_id: int
    size: int
    name: str
    definition: str
    coherent: bool
    samples: list[str]


def openings(conversations: list[Conversation]) -> list[str]:
    return [c.opening["text"] for c in conversations]


def cluster(vectors: np.ndarray, k: int = N_CLUSTERS) -> np.ndarray:
    return KMeans(n_clusters=k, random_state=config.SEED, n_init=10).fit_predict(vectors)


def _representatives(
    vectors: np.ndarray, texts: list[str], labels: np.ndarray, cluster_id: int, n: int
) -> list[str]:
    members = np.flatnonzero(labels == cluster_id)
    centre = vectors[members].mean(axis=0)
    # Nearest to the centroid, so the LLM names the theme rather than an outlier.
    closest = members[np.argsort(-(vectors[members] @ centre))][:n]
    return [texts[i] for i in closest]


def induce(conversations: list[Conversation], k: int = N_CLUSTERS) -> list[Cluster]:
    texts = openings(conversations)
    vectors = embed.embed(texts, embed.CLUSTERING)
    labels = cluster(vectors, k)

    out = []
    for cid in range(k):
        samples = _representatives(vectors, texts, labels, cid, SAMPLES_PER_CLUSTER)
        listed = "\n".join(f"- {s}" for s in samples)
        result = client.complete(
            config.AGENT_MODEL,
            PROPOSE_SYSTEM,
            f"Customer messages in this group:\n{listed}",
            schema=PROPOSE_SCHEMA,
        )
        proposal = json.loads(result.text)
        out.append(
            Cluster(
                cluster_id=cid,
                size=int((labels == cid).sum()),
                name=proposal["name"],
                definition=proposal["definition"],
                coherent=proposal["coherent"],
                samples=samples,
            )
        )
    return sorted(out, key=lambda c: -c.size)
