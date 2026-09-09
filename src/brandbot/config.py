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

# Measured 3/3 availability at 1.6s per call on the free tier, where 3.8-flash and
# flash-latest returned 503 and 3.6-flash took 21s. A judge that silently swaps models
# mid-run produces verdicts that are not comparable, so this one never falls back:
# an unservable judge call raises instead.
JUDGE_MODEL = "gemini-3.7-flash"

EMBED_MODEL = "gemini-embedding-001"

EMBED_DIM = 256

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def groq_key() -> str | None:
    return os.environ.get("GROQ_API_KEY") or None


def gemini_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or None
