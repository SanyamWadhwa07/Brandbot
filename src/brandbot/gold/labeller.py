"""Render the offline labelling page.

The page is a single self-contained file so labelling needs no server, no
network and no keys, and so the exact page a label was written on can be
committed next to the label it produced.

The stratum is not passed through. A labeller who can see that a message was
drawn to top up a thin intent has been told what to expect, and the top-up
signal would leak into the label it was supposed to stay independent of.
"""

from __future__ import annotations

import json
from pathlib import Path

from brandbot import config
from brandbot.gold import spec
from brandbot.gold.sample import Candidate
from brandbot.intents import taxonomy

TEMPLATE = config.ROOT / "tools" / "label.html"
PLACEHOLDER = "__DATA__"


def payload(
    candidates: list[Candidate], set_name: str, prefill: dict[str, dict] | None = None
) -> dict:
    return {
        "set_name": set_name,
        # Pre-filled labels the reviewer accepts or overrides. Whether each one was
        # changed is recorded: an adjudication pass where nothing changes is a
        # rubber stamp, and the override rate is the only evidence of the difference.
        "prefill": prefill or {},
        "intents": [
            {"name": i.name, "definition": f"{i.definition} Not: {i.excludes}"}
            for i in taxonomy.TAXONOMY
        ]
        + [{"name": taxonomy.OTHER, "definition": "None of the above fits."}],
        "routes": [{"code": r.code, "key": r.key, "definition": r.definition} for r in spec.ROUTES],
        "asks": [{"code": a.code, "key": a.key, "definition": a.definition} for a in spec.ASK_TYPES],
        # Shown on the page so a decision made at example 200 is made the same way
        # it was at example 3, without relying on the labeller's memory.
        "rules": list(spec.TIE_BREAKS),
        "examples": [
            {
                "id": c.id,
                "thread_id": c.thread_id,
                "text": c.text,
                "turns": [{"role": t["role"], "text": t["text"]} for t in c.turns],
            }
            for c in candidates
        ],
    }


def render(
    candidates: list[Candidate], out: Path, set_name: str, prefill: dict[str, dict] | None = None
) -> Path:
    data = json.dumps(payload(candidates, set_name, prefill), ensure_ascii=False)
    # The payload sits in a <script type="application/json"> block, so the only
    # sequence that can break out of it is a literal closing script tag.
    html = TEMPLATE.read_text(encoding="utf-8").replace(
        PLACEHOLDER, data.replace("</", "<\\/")
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out
