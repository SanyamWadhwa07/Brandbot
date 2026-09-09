"""Streaming parse of the raw corpus into a compact reply graph.

The raw CSV is 3M rows and this machine has ~2GB free, so tweet text is
deliberately left on disk here. Pass one builds a numpy index (~90MB) that is
enough to reconstruct threads and score brands; text is fetched later for the
one brand that gets selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from brandbot import config

TWEET_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"
CHUNK_ROWS = 250_000

INDEX_PATH = config.DATA / "interim" / "reply_graph.npz"


@dataclass(frozen=True)
class ReplyGraph:
    """Every tweet in the corpus, minus its text."""

    tweet_id: np.ndarray  # int64
    author: np.ndarray  # int32, indexes into `authors`
    authors: list[str]
    inbound: np.ndarray  # bool: True when written by a customer
    created_at: np.ndarray  # int64 epoch seconds, UTC
    parent: np.ndarray  # int64 tweet_id replied to, -1 when the tweet starts a thread

    def __len__(self) -> int:
        return len(self.tweet_id)

    def author_names(self) -> np.ndarray:
        return np.array(self.authors, dtype=object)[self.author]


def build_reply_graph(csv_path: Path | None = None) -> ReplyGraph:
    csv_path = csv_path or config.RAW / "twcs.csv"

    ids, authors_codes, inbound, created, parent = [], [], [], [], []
    author_index: dict[str, int] = {}

    reader = pd.read_csv(
        csv_path,
        usecols=["tweet_id", "author_id", "inbound", "created_at", "in_response_to_tweet_id"],
        dtype={"tweet_id": "int64", "author_id": "string", "inbound": "bool"},
        chunksize=CHUNK_ROWS,
    )
    for chunk in reader:
        ids.append(chunk["tweet_id"].to_numpy(np.int64))
        inbound.append(chunk["inbound"].to_numpy(bool))

        codes = np.empty(len(chunk), dtype=np.int32)
        for i, name in enumerate(chunk["author_id"].to_numpy()):
            code = author_index.get(name)
            if code is None:
                code = len(author_index)
                author_index[name] = code
            codes[i] = code
        authors_codes.append(codes)

        # pandas infers the datetime resolution (us here, ns on older versions), so
        # cast through an explicit second-resolution dtype rather than dividing.
        ts = pd.to_datetime(chunk["created_at"], format=TWEET_DATE_FORMAT, utc=True)
        created.append(ts.dt.tz_convert(None).values.astype("datetime64[s]").astype(np.int64))

        # NaN means the tweet opens a thread rather than replying to one.
        par = chunk["in_response_to_tweet_id"].to_numpy(dtype="float64")
        parent.append(np.where(np.isnan(par), -1, par).astype(np.int64))

    return ReplyGraph(
        tweet_id=np.concatenate(ids),
        author=np.concatenate(authors_codes),
        authors=list(author_index),
        inbound=np.concatenate(inbound),
        created_at=np.concatenate(created),
        parent=np.concatenate(parent),
    )


def save(graph: ReplyGraph, path: Path = INDEX_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        tweet_id=graph.tweet_id,
        author=graph.author,
        authors=np.array(graph.authors, dtype=object),
        inbound=graph.inbound,
        created_at=graph.created_at,
        parent=graph.parent,
    )


def load(path: Path = INDEX_PATH) -> ReplyGraph:
    with np.load(path, allow_pickle=True) as z:
        return ReplyGraph(
            tweet_id=z["tweet_id"],
            author=z["author"],
            authors=list(z["authors"]),
            inbound=z["inbound"],
            created_at=z["created_at"],
            parent=z["parent"],
        )


def load_text(rows: np.ndarray, csv_path: Path | None = None) -> dict[int, str]:
    """Tweet text for specific row indices, read without holding the corpus in memory.

    Chunks arrive in file order, so a chunk's row indices are known from its position
    and the wanted rows can be sliced out as it streams past.
    """
    csv_path = csv_path or config.RAW / "twcs.csv"
    wanted = np.sort(np.asarray(rows, dtype=np.int64))
    out: dict[int, str] = {}

    reader = pd.read_csv(csv_path, usecols=["text"], dtype={"text": "string"}, chunksize=CHUNK_ROWS)
    for i, chunk in enumerate(reader):
        lo, hi = i * CHUNK_ROWS, i * CHUNK_ROWS + len(chunk)
        take = wanted[(wanted >= lo) & (wanted < hi)]
        if len(take) == 0:
            continue
        values = chunk["text"].to_numpy()
        for r in take:
            out[int(r)] = values[r - lo]
    return out
