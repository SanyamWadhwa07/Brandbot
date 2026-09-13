"""Pick the replies a human reviews, so the judge can be checked against a person.

An LLM judge is an instrument, and an instrument nobody calibrated is decoration.
The only way to know whether its verdicts mean anything is to have a human grade a
sample blind and measure whether the two agree more than chance explains.

Three rules make the comparison worth anything.

The sample is drawn at random inside each system, never by verdict. Choosing
replies the judge was confident about, or ones it failed, selects on the very thing
being measured and would produce an agreement figure that says nothing.

The rater sees the message, the reply and the precedents, and nothing else. Not
which system wrote it, not what the judge said. Both would anchor the answer.

Systems are sampled in roughly equal numbers rather than in proportion. The trivial
baseline sends one identical line to every message, so a proportional draw would
spend most of the human's attention re-reading the same sentence. Agreement is
reported per system as well as overall, because a judge can be reliable on obvious
failures and useless on the close ones.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from brandbot import config
from brandbot.data import brand_slice, splits
from brandbot.eval.judge import DIMENSIONS
from brandbot.eval.run import Run
from brandbot.gold import spec
from brandbot.pipeline import agent
from brandbot.retrieval import index

TEMPLATE = config.ROOT / "tools" / "rate.html"
PLACEHOLDER = "__DATA__"
PER_SYSTEM = 27

DEFINITIONS = {
    "grounded": "says something the past replies do not support",
    "addresses": "answers a different question than the one asked",
    "safe": "promises something the brand would be held to",
    "tone": "does not sound like this brand",
}


@dataclass(frozen=True)
class Sampled:
    id: str
    thread_id: int
    system: str
    message: str
    reply: str


def draw(runs: list[Run], *, per_system: int = PER_SYSTEM, seed: int = config.SEED) -> list[Sampled]:
    rng = random.Random(seed)
    out: list[Sampled] = []
    for run in runs:
        sendable = [h for h in run.handlings if h.route == spec.AUTO and h.reply]
        for h in rng.sample(sendable, min(per_system, len(sendable))):
            out.append(
                Sampled(
                    id=f"{run.system}-{h.thread_id}",
                    thread_id=h.thread_id,
                    system=run.system,
                    message=h.message,
                    reply=h.reply,
                )
            )
    rng.shuffle(out)
    return out


def render(sampled: list[Sampled], out: Path, set_name: str) -> Path:
    parts = splits.partition(brand_slice.load())
    ix = index.build(parts[splits.Split.INDEX])

    payload = {
        "set_name": set_name,
        "dimensions": [{"name": d, "definition": DEFINITIONS[d]} for d in DIMENSIONS],
        # `system` is deliberately absent: the rater must not know whose reply it is.
        "examples": [
            {
                "id": s.id,
                "thread_id": s.thread_id,
                "message": s.message,
                "reply": s.reply,
                "precedents": [
                    {"message": h.precedent.message, "reply": h.precedent.reply}
                    for h in ix.search(s.message, agent.K)
                ],
            }
            for s in sampled
        ],
    }
    data = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(TEMPLATE.read_text(encoding="utf-8").replace(PLACEHOLDER, data), encoding="utf-8")
    return out


def key(sampled: list[Sampled], path: Path) -> Path:
    """Which system wrote each sampled reply, written where the rater cannot see it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in sampled:
            f.write(json.dumps({"id": s.id, "system": s.system, "thread_id": s.thread_id}) + "\n")
    return path
