"""Paths, seeds and model identifiers. Single source of truth for anything
a run needs to be reproducible."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
BRAND = DATA / "brand"
GOLD = DATA / "gold"
ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"

SEED = 20260909

# Groq enforces rate limits per model ID rather than per account, so spreading
# roles across IDs multiplies the usable free-tier budget.
AGENT_MODEL = "openai/gpt-oss-120b"
AGENT_SMALL_MODEL = "openai/gpt-oss-20b"
CROSS_JUDGE_MODEL = "qwen/qwen3.8-27b"
LOCAL_MODEL = "qwen2.5:7b-instruct-q4_K_M"

# Chosen on two measurements, not on version number. Free-tier capacity is reserved
# for the lite models: over 6 attempts each, every full flash model returned 503
# (3.7-flash managed 3/6, and 0/8 an hour later) while every lite model went 6/6.
# On 8 hand-built cases with known answers, 3.1-flash-lite scored 7/8 on
# groundedness against 5/8 for the newer 3.5-flash-lite.
#
# A judge that silently swaps models mid-run mixes two graders into one score and
# quietly invalidates the agreement statistics, so this one never falls back: an
# unservable judge call raises. Aliases like `gemini-flash-lite-latest` are avoided
# for the same reason, since the model behind them can change under a pinned name.
JUDGE_MODEL = "gemini-3.1-flash-lite"

# Local, via Ollama. 51 texts/sec against Gemini free tier's 1.7, no quota, no
# key, and wider semantic separation on this corpus. See retrieval/embed.py.
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def groq_key() -> str | None:
    return os.environ.get("GROQ_API_KEY") or None


def gemini_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or None
