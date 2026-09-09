"""Local model, for the on-prem ablation row.

Measured at ~25s per call on this machine against ~1s for the hosted models, so
it is a one-shot ablation rather than anything used during development.
"""

from __future__ import annotations

import json
import urllib.request

from brandbot import config
from brandbot.llm.cache import Call, Result

TIMEOUT_SECONDS = 600


def complete(call: Call) -> Result:
    body = {
        "model": call.model,
        "messages": [
            {"role": "system", "content": call.system},
            {"role": "user", "content": call.user},
        ],
        "stream": False,
        "options": {"temperature": call.temperature, "seed": config.SEED},
    }
    if call.schema is not None:
        body["format"] = call.schema

    req = urllib.request.Request(
        f"{config.OLLAMA_HOST}/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        payload = json.load(resp)
    return Result(
        text=payload["message"]["content"],
        prompt_tokens=payload.get("prompt_eval_count", 0),
        completion_tokens=payload.get("eval_count", 0),
    )
