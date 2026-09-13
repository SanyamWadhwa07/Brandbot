"""Materialise one brand's conversations as the committed working corpus.

The raw dump is 516MB and gitignored; this slice is ~2MB and ships with the repo
so the pipeline is reproducible without a Kaggle account. The brief explicitly
sanctions working from a subsample.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from brandbot import config
from brandbot.data import ingest
from brandbot.data.ingest import ReplyGraph
from brandbot.data.threads import Threads

SLICE_PATH = config.BRAND / "threads.jsonl.gz"
FOREIGN_PATH = config.BRAND / "foreign_replies.json"

LEADING_MENTIONS = re.compile(r"^(?:@\w+\s+)+")
EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
URL = re.compile(r"https?://\S+")
# Order and account numbers. Short runs are left alone: they are usually years,
# prices or ZIP codes, and destroying those would cost more than it protects.
LONG_DIGITS = re.compile(r"\b\d{7,}\b")
WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Turn:
    tweet_id: int
    role: str  # "customer" or "brand"
    created_at: int  # epoch seconds
    text: str


@dataclass(frozen=True)
class Conversation:
    thread_id: int
    brand: str
    started_at: int
    turns: list[dict]

    @property
    def opening(self) -> dict:
        return self.turns[0]


def clean(text: str) -> str:
    """Strip routing noise and obvious identifiers, leaving the wording intact.

    Links become a placeholder rather than being dropped: the reply still reads as
    pointing somewhere, but the model cannot invent a plausible-looking URL and
    have it counted as grounded.
    """
    text = LEADING_MENTIONS.sub("", text)
    text = EMAIL.sub("<email>", text)
    text = URL.sub("<link>", text)
    text = LONG_DIGITS.sub("<number>", text)
    return WHITESPACE.sub(" ", text).strip()


def extract(
    graph: ReplyGraph, table: Threads, roots: np.ndarray, brand: str
) -> list[Conversation]:
    code = graph.authors.index(brand)
    keep = table.is_conversation() & (table.brand == code)
    wanted_roots = table.root_row[keep]

    members = np.flatnonzero(np.isin(roots, wanted_roots))
    text = ingest.load_text(members)
    ordered = members[np.lexsort((graph.created_at[members], roots[members]))]

    grouped: dict[int, list[Turn]] = {}
    for row in ordered:
        grouped.setdefault(int(roots[row]), []).append(
            Turn(
                tweet_id=int(graph.tweet_id[row]),
                role="customer" if graph.inbound[row] else "brand",
                created_at=int(graph.created_at[row]),
                text=clean(text[int(row)]),
            )
        )

    out = [
        Conversation(
            thread_id=turns[0].tweet_id,
            brand=brand,
            started_at=turns[0].created_at,
            turns=[asdict(t) for t in turns],
        )
        for turns in grouped.values()
        if turns[0].role == "customer" and any(t.role == "brand" for t in turns)
    ]
    return sorted(out, key=lambda c: (c.started_at, c.thread_id))


def foreign_reply_threads(
    graph: ReplyGraph, table: Threads, roots: np.ndarray, brand: str
) -> list[int]:
    """Threads where a company other than `brand` also replied.

    The corpus marks every company tweet outbound without recording which company
    wrote it, so a customer who tagged three support accounts leaves another team's
    words in this brand's thread, indistinguishable from the brand's own. It is
    rare, 41 threads of 14,122, and the damage is concentrated: when the foreign
    reply lands first it becomes the precedent for that thread, and a draft
    grounded in it answers in the wrong company's voice.

    Recording the ids rather than dropping the turns keeps the committed slice
    byte-identical, which matters while a golden set is being labelled against it.
    """
    code = graph.authors.index(brand)
    keep = table.is_conversation() & (table.brand == code)
    members = np.flatnonzero(np.isin(roots, table.root_row[keep]))
    outbound = members[~graph.inbound[members]]
    foreign = outbound[graph.author[outbound] != code]
    return sorted({int(graph.tweet_id[int(roots[row])]) for row in foreign})


def load_foreign(path: Path = FOREIGN_PATH) -> set[int]:
    return set(json.loads(path.read_text(encoding="utf-8")))


def save(conversations: list[Conversation], path: Path = SLICE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for c in conversations:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


def load(path: Path = SLICE_PATH) -> list[Conversation]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [Conversation(**json.loads(line)) for line in f]
