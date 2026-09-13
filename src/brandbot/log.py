"""One audit trail for every model call and every routing decision.

Two readers, and they want different things from the same record.

An operator needs to answer "why did this customer get this reply" months later.
For a system that sends replies without a human seeing them first, that is not a
convenience: the decision, the reason, the precedents it leaned on and the model
that wrote it all have to be recoverable from the record alone.

Whoever is running a batch needs to see it moving. Long runs against a metered
free tier spend most of their time asleep, and a silent process is
indistinguishable from a wedged one. That distinction cost real time here before
this existed.

Events go to a JSON Lines file, one object per line, because the run log is data
to be counted and filtered rather than prose to be read. A short human-readable
line goes to stderr at the same time, so a run in a terminal shows progress
without anyone tailing a file.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from brandbot import config

RUN_LOG = config.ARTIFACTS / "run.jsonl"

_events: Path | None = None


def setup(path: Path = RUN_LOG, *, level: int = logging.INFO, echo: bool = True) -> None:
    global _events
    path.parent.mkdir(parents=True, exist_ok=True)
    _events = path

    root = logging.getLogger("brandbot")
    root.setLevel(level)
    root.handlers.clear()
    if echo:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        root.addHandler(handler)


def event(kind: str, **fields) -> None:
    """Record one structured fact. Silently does nothing until `setup` has run.

    No timestamp is written into the record. Run logs are compared across runs and
    a clock reading makes two otherwise identical runs differ, which is the same
    reason nothing hashed or cached is allowed to call the clock.
    """
    if _events is None:
        return
    with _events.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": kind, **fields}, ensure_ascii=False) + "\n")


def get(name: str) -> logging.Logger:
    return logging.getLogger(f"brandbot.{name}")


@contextmanager
def progress(logger: logging.Logger, label: str, total: int, every: int = 20):
    """Report how far a batch has got, and how fast, without flooding the terminal."""
    started = time.monotonic()
    state = {"done": 0}

    def tick() -> None:
        state["done"] += 1
        n = state["done"]
        if n % every == 0 or n == total:
            rate = n / max(time.monotonic() - started, 1e-9)
            remaining = (total - n) / rate if rate else 0
            logger.info(f"{label} {n}/{total} ({rate * 60:.0f}/min, ~{remaining / 60:.0f} min left)")

    yield tick
    logger.info(f"{label} finished {state['done']} in {(time.monotonic() - started) / 60:.1f} min")
