"""Embeddings, computed locally.

Gemini's embedding endpoint was tried first and abandoned. The free tier meters
per text at 100 a minute, so the 13,628-precedent index would have taken about
2.3 hours and stalled repeatedly on 429s. `nomic-embed-text` under Ollama does
the same work in about 4 minutes with no quota, no key, and better separation on
this data: for "hulu keeps buffering" against "cancel my subscription", Gemini
returned 0.772 and nomic returns 0.453, while nomic still scores the genuinely
related pair higher.

Vectors are cached on disk because re-embedding unchanged text is pure waste.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np

from brandbot import config

CACHE_DIR = config.DATA / "interim" / "embeddings"
BATCH = 256
TIMEOUT_SECONDS = 600

# nomic-embed-text is trained with task prefixes and loses accuracy without them.
DOCUMENT = "search_document: "
QUERY = "search_query: "
CLUSTERING = "clustering: "


def _key(text: str, task: str) -> str:
    payload = json.dumps([config.EMBED_MODEL, task, text], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _call(texts: list[str]) -> np.ndarray:
    request = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/embed",
        data=json.dumps({"model": config.EMBED_MODEL, "input": texts}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        vectors = np.array(json.load(response)["embeddings"], dtype=np.float32)
    # Normalised here so retrieval is a plain dot product everywhere downstream.
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def embed(texts: list[str], task: str, cache_dir: Path = CACHE_DIR) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    keys = [_key(t, task) for t in texts]

    vectors: list[np.ndarray | None] = []
    missing: list[int] = []
    for i, k in enumerate(keys):
        path = cache_dir / f"{k}.npy"
        if path.exists():
            vectors.append(np.load(path))
        else:
            vectors.append(None)
            missing.append(i)

    for start in range(0, len(missing), BATCH):
        chunk = missing[start : start + BATCH]
        fresh = _call([task + texts[i] for i in chunk])
        for i, vector in zip(chunk, fresh, strict=True):
            np.save(cache_dir / f"{keys[i]}.npy", vector)
            vectors[i] = vector

    if not vectors:
        return np.empty((0, config.EMBED_DIM), dtype=np.float32)
    return np.stack(vectors)
