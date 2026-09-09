"""Gemini: the primary judge, and the embedding model behind retrieval.

The judge sits outside the family of every model it grades, so a reply cannot be
rewarded for looking like something the judge would have written itself.
"""

from __future__ import annotations

import numpy as np
from google import genai
from google.genai import types

from brandbot import config
from brandbot.llm.cache import Call, Result

# Gemini accepts an OpenAPI 3.0 subset, not full JSON Schema. Groq's strict mode
# demands "additionalProperties": false, which Gemini rejects outright, so one
# canonical schema is written per call site and narrowed here.
GEMINI_SCHEMA_KEYS = frozenset(
    {"type", "format", "description", "nullable", "enum", "items", "properties",
     "required", "minItems", "maxItems", "anyOf", "propertyOrdering"}
)


def to_gemini_schema(node: object) -> object:
    if isinstance(node, dict):
        out: dict = {}
        for k, v in node.items():
            if k not in GEMINI_SCHEMA_KEYS:
                continue
            # Keys under "properties" are field names, not schema keywords. Filtering
            # them against the keyword list deletes the fields themselves and leaves
            # "required" pointing at properties that no longer exist.
            out[k] = (
                {name: to_gemini_schema(sub) for name, sub in v.items()}
                if k == "properties"
                else to_gemini_schema(v)
            )
        return out
    if isinstance(node, list):
        return [to_gemini_schema(v) for v in node]
    return node

_client: genai.Client | None = None


def client() -> genai.Client:
    global _client
    if _client is None:
        key = config.gemini_key()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is unset; live calls need it, `--replay` does not")
        _client = genai.Client(api_key=key)
    return _client


def complete(call: Call) -> Result:
    cfg = types.GenerateContentConfig(
        system_instruction=call.system or None,
        temperature=call.temperature,
        **(
            {
                "response_mime_type": "application/json",
                "response_schema": to_gemini_schema(call.schema),
            }
            if call.schema is not None
            else {}
        ),
    )
    r = client().models.generate_content(model=call.model, contents=call.user, config=cfg)
    usage = r.usage_metadata
    return Result(
        text=r.text or "",
        prompt_tokens=usage.prompt_token_count or 0,
        completion_tokens=(usage.candidates_token_count or 0),
    )


def embed(texts: list[str], task: str) -> np.ndarray:
    """L2-normalised embeddings, truncated to `config.EMBED_DIM`.

    Truncation is Matryoshka-style and supported by the model, which keeps the
    committed index small enough to ship in the repo. Normalising here means
    retrieval is a plain dot product with no per-query rescaling.
    """
    r = client().models.embed_content(
        model=config.EMBED_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            output_dimensionality=config.EMBED_DIM, task_type=task
        ),
    )
    vecs = np.array([e.values for e in r.embeddings], dtype=np.float32)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
