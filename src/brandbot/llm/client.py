"""One entry point for every model call.

Responsibilities in order: serve from cache, refuse to go live when replaying,
retry transient failures, validate structured output, and record spend.
"""

from __future__ import annotations

import json
import random
import re
import time
from pathlib import Path

from brandbot import config
from brandbot.llm import cache
from brandbot.llm.budget import Budget
from brandbot.llm.cache import Call, Result
from brandbot.llm.providers import gemini_provider, groq_provider, ollama_provider

# Free-tier capacity comes and goes in minutes, not seconds: a model measured at
# 6/6 availability can return 503 "high demand" an hour later. Batch jobs can
# afford to wait, so the transient budget is generous and capped per sleep.
TRANSIENT_ATTEMPTS = 8
MAX_BACKOFF_SECONDS = 60.0
BASE_BACKOFF_SECONDS = 2.0

# A schema violation is a sampling accident, not congestion. Resample immediately.
SCHEMA_ATTEMPTS = 3

TRANSIENT = ("429", "rate limit", "503", "unavailable", "overloaded", "500", "timeout")

# Groq meters tokens per minute and says exactly how long to wait: "Please try
# again in 6.394s". Guessing with exponential backoff instead wastes most of the
# window, because a 2-second retry into a 60-second bucket fails again and the
# doubling then overshoots. Measured on this workload the guess ran at about two
# calls a minute against a ceiling of eleven.
RETRY_AFTER = re.compile(r"try again in (?:(\d+)m)?([\d.]+)s", re.I)


def _requested_wait(exc: Exception) -> float | None:
    match = RETRY_AFTER.search(str(exc))
    if not match:
        return None
    minutes, seconds = match.group(1), match.group(2)
    return (int(minutes) * 60 if minutes else 0) + float(seconds)


class ReplayMiss(RuntimeError):
    """A replay run needed a response that was never recorded."""


class SchemaViolation(RuntimeError):
    """The model returned something its own schema forbids."""


def _provider(model: str):
    if model.startswith("gemini"):
        return "gemini", gemini_provider
    # Groq serves namespaced open weights ("openai/...", "qwen/..."); Ollama tags
    # use a colon ("qwen2.5:7b-instruct-q4_K_M").
    if "/" in model:
        return "groq", groq_provider
    return "ollama", ollama_provider


def _is_transient(exc: Exception) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in TRANSIENT)


def complete(
    model: str,
    system: str,
    user: str,
    *,
    schema: dict | None = None,
    temperature: float = 0.0,
    extra: dict | None = None,
    budget: Budget | None = None,
    replay_only: bool = False,
    cache_root: Path = cache.CACHE_DIR,
) -> Result:
    name, module = _provider(model)
    call = Call(name, model, system, user, schema, temperature, extra or {})

    hit = cache.get(call, cache_root)
    if hit is not None:
        if budget:
            budget.record(model, hit.prompt_tokens + hit.completion_tokens, was_cached=True)
        return hit

    if replay_only:
        raise ReplayMiss(f"{model}: no recorded response for this prompt ({cache.key(call)[:12]})")

    last: Exception | None = None
    transient_used = schema_used = 0

    while transient_used < TRANSIENT_ATTEMPTS and schema_used < SCHEMA_ATTEMPTS:
        try:
            result = module.complete(call)
            if schema is not None:
                _validate(result.text, model)
            cache.put(call, result, cache_root)
            if budget:
                budget.record(model, result.prompt_tokens + result.completion_tokens, False)
            return result
        except SchemaViolation as exc:
            # Structured output is documented as guaranteed but has been reported to
            # slip on gpt-oss, so a violation is resampled rather than trusted.
            last, schema_used = exc, schema_used + 1
            continue
        except Exception as exc:
            if not _is_transient(exc):
                raise
            last, transient_used = exc, transient_used + 1
        asked = _requested_wait(last) if last else None
        time.sleep(
            min(
                asked
                if asked is not None
                else BASE_BACKOFF_SECONDS * 2 ** (transient_used - 1),
                MAX_BACKOFF_SECONDS,
            )
            + random.random()
        )

    raise RuntimeError(
        f"{model} gave up after {transient_used} transient and {schema_used} schema retries: {last}"
    ) from last


def _validate(text: str, model: str) -> None:
    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaViolation(f"{model} returned non-JSON under a schema: {text[:120]!r}") from exc


def judge(user: str, system: str, schema: dict, **kwargs) -> Result:
    """Every judge verdict comes from one pinned model.

    Falling back to a different judge mid-run would mix two graders' opinions into
    one score, and the agreement statistics would silently stop meaning anything.
    """
    return complete(config.JUDGE_MODEL, system, user, schema=schema, **kwargs)
