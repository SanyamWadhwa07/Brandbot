"""Groq-hosted open-weight models: the agent, and the cross-vendor second judge."""

from __future__ import annotations

import json

from groq import Groq

from brandbot import config
from brandbot.llm.cache import Call, Result

_client: Groq | None = None


def client() -> Groq:
    global _client
    if _client is None:
        key = config.groq_key()
        if not key:
            raise RuntimeError("GROQ_API_KEY is unset; live calls need it, `--replay` does not")
        _client = Groq(api_key=key)
    return _client


def complete(call: Call) -> Result:
    kwargs = {}
    if call.schema is not None:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": call.schema, "strict": True},
        }
    # gpt-oss emits reasoning tokens that count against the daily cap. At the
    # default effort a short classification answer costs ~3x what it does here.
    if "gpt-oss" in call.model:
        kwargs["reasoning_effort"] = call.extra.get("reasoning_effort", "low")

    r = client().chat.completions.create(
        model=call.model,
        messages=[
            {"role": "system", "content": call.system},
            {"role": "user", "content": call.user},
        ],
        temperature=call.temperature,
        **kwargs,
    )
    return Result(
        text=r.choices[0].message.content or "",
        prompt_tokens=r.usage.prompt_tokens,
        completion_tokens=r.usage.completion_tokens,
    )
