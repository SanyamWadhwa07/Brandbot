"""Draw the golden-set candidates from the gold split.

Two strata, because one sample cannot answer both questions the report needs.

`RANDOM` is drawn uniformly, so the intent mix inside it is the mix real traffic
has. It is the only honest source for the prior that `metrics.prior_weighted_accuracy`
reweights by, and it is what makes "how would this do on tomorrow's stream" answerable.

`ENRICHED` tops up the intents the uniform draw leaves too thin to measure. At this
brand's mix, a 150-message uniform sample gives the rarest intents about eight
examples each, and a per-class F1 on eight examples has a confidence interval wide
enough to be useless. Enrichment buys measurable rare classes at the cost of a
label distribution that no longer matches traffic, which is exactly why the two
strata stay separable in the output.

The enrichment signal is the nearest intent *definition* by embedding. It chooses
which messages get shown to the labeller and never what label they receive: the
stratum is not rendered in the tool, the labeller cannot see it, and every label
is written by hand against the taxonomy text. Recorded here rather than buried,
because a reviewer is entitled to check that the selection signal and the label
came from different places.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from brandbot import config
from brandbot.data import brand_slice, splits
from brandbot.data.brand_slice import Conversation
from brandbot.intents import induce, merge, taxonomy
from brandbot.retrieval import embed

# The sampling record is interim, not gold. `data/gold/` holds only what a human
# wrote; keeping the strata out of it means no field in the labelled file was
# derived from a model, and the join happens at load time instead.
CANDIDATES_PATH = config.DATA / "interim" / "gold_candidates.jsonl"

N_RANDOM = 150
N_ENRICHED = 70

# Below this a message carries no intent to recover ("@hulu_support ?"), and the
# labeller would be guessing rather than labelling.
MIN_CHARS = 15

RANDOM = "random"
ENRICHED = "enriched"


@dataclass(frozen=True)
class Candidate:
    id: str
    thread_id: int
    text: str
    stratum: str
    turns: list[dict]


def eligible(conversations: list[Conversation]) -> list[Conversation]:
    return [c for c in conversations if len(c.opening["text"]) >= MIN_CHARS]


def nearest_intent(texts: list[str], dev: list[Conversation]) -> list[str]:
    """Assign each message to an intent through the frozen dev clustering.

    Matching a message against the *definition* of an intent was tried first and
    is much worse: embedding models compare messages to messages well and messages
    to abstract category prose badly, and on this pool it put 87 of 220 messages
    under `watchlist_issue`, whose definition happens to sit near the centre of
    everything. Cluster centroids are averaged real messages, so the comparison
    stays message-to-message.

    Refitting is deterministic under a fixed seed and the dev embeddings are
    cached, so this reproduces the clustering that the merge map describes.
    """
    centroids = _dev_centroids(dev)
    intents = np.array([merge.CLUSTER_MERGE[c] for c in range(induce.N_CLUSTERS)])
    vectors = embed.embed(texts, embed.CLUSTERING)
    return list(intents[np.argmax(vectors @ centroids.T, axis=1)])


def _dev_centroids(dev: list[Conversation]) -> np.ndarray:
    vectors = embed.embed([c.opening["text"] for c in dev], embed.CLUSTERING)
    labels = induce.cluster(vectors)
    centroids = np.stack([vectors[labels == c].mean(axis=0) for c in range(induce.N_CLUSTERS)])
    return centroids / np.linalg.norm(centroids, axis=1, keepdims=True)


def _top_up(
    pools: dict[str, list[int]], taken: set[int], counts: dict[str, int], n: int, rng: random.Random
) -> list[tuple[int, str]]:
    """Repeatedly feed the thinnest intent that still has candidates left.

    Balancing against a running count rather than a fixed per-intent quota means
    an intent the uniform draw already covered well does not get topped up at the
    expense of one it missed.
    """
    out: list[tuple[int, str]] = []
    for _ in range(n):
        available = [name for name, pool in pools.items() if any(i not in taken for i in pool)]
        if not available:
            break
        target = min(available, key=lambda name: (counts[name], name))
        choice = rng.choice([i for i in pools[target] if i not in taken])
        taken.add(choice)
        counts[target] += 1
        out.append((choice, target))
    return out


def draw(
    conversations: list[Conversation],
    dev: list[Conversation],
    *,
    n_random: int = N_RANDOM,
    n_enriched: int = N_ENRICHED,
    seed: int = config.SEED,
) -> list[Candidate]:
    pool = eligible(conversations)
    rng = random.Random(seed)

    chosen = set(rng.sample(range(len(pool)), n_random))
    assigned = nearest_intent([c.opening["text"] for c in pool], dev)

    by_intent: dict[str, list[int]] = {name: [] for name in taxonomy.names()}
    for i, name in enumerate(assigned):
        by_intent[name].append(i)

    counts = {name: 0 for name in by_intent}
    for i in chosen:
        counts[assigned[i]] += 1

    strata = {i: RANDOM for i in chosen}
    for i, _ in _top_up(by_intent, chosen, counts, n_enriched, rng):
        strata[i] = ENRICHED

    # Shuffled so the labeller never meets a run of one intent, which is where
    # anchoring creeps in and where a fatigued rater starts pattern-matching.
    order = sorted(strata)
    rng.shuffle(order)

    return [
        Candidate(
            id=str(pool[i].thread_id),
            thread_id=pool[i].thread_id,
            text=pool[i].opening["text"],
            stratum=strata[i],
            turns=pool[i].turns,
        )
        for i in order
    ]


def build(path: Path) -> list[Candidate]:
    parts = splits.partition(brand_slice.load())
    candidates = draw(parts[splits.Split.GOLD], parts[splits.Split.DEV])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    return candidates
