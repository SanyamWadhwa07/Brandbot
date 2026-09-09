"""On-disk cache of every model response, keyed by exactly what was sent.

Two jobs. During development it makes a rate-limited free tier survivable: a run
that dies at 429 resumes without repaying for the calls it already made. After
that it is what makes `eval --replay` work, so a reviewer can regenerate every
headline number with no keys and no network.

Entries are committed. The key covers the model and the full rendered prompt, so
a cached entry can never correspond to a prompt other than the one that produced
it, and editing a prompt silently misses rather than returning a stale answer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from brandbot import config

CACHE_DIR = config.ARTIFACTS / "cache"


@dataclass(frozen=True)
class Call:
    provider: str
    model: str
    system: str
    user: str
    schema: dict | None
    temperature: float
    extra: dict


@dataclass(frozen=True)
class Result:
    text: str
    prompt_tokens: int
    completion_tokens: int


def key(call: Call) -> str:
    payload = json.dumps(asdict(call), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def path_for(k: str, root: Path = CACHE_DIR) -> Path:
    # Sharded so no directory holds thousands of entries.
    return root / k[:2] / f"{k}.json"


def get(call: Call, root: Path = CACHE_DIR) -> Result | None:
    p = path_for(key(call), root)
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return Result(**json.load(f)["result"])


def put(call: Call, result: Result, root: Path = CACHE_DIR) -> None:
    p = path_for(key(call), root)
    p.parent.mkdir(parents=True, exist_ok=True)
    # The call is stored alongside the result so an entry is auditable on its own.
    with p.open("w", encoding="utf-8") as f:
        json.dump({"call": asdict(call), "result": asdict(result)}, f, ensure_ascii=False, indent=1)
