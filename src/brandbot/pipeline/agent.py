"""One inbound message in, one handling decision out.

Three stages in a fixed order: label the message, decide whether a human has to
take it, and only then draft. The order is what keeps the cost down and the
measurement clean.

Classification sees the message alone, never the retrieved precedents. Intent
accuracy then measures classification rather than retrieval quality leaking in
through the prompt, and the number stays comparable to the zero-retrieval
baseline.

Drafting runs only on messages the policy gate lets through. Escalated messages
never reach a second model call, which on this brand's mix is most of them.

`confidence` is the model's own estimate and self-reported confidence is badly
calibrated almost everywhere. It is emitted rather than trusted: the report
ranks it against two signals that cost nothing, the top precedent's score and
its margin over the runner-up, and lets the risk-coverage curve decide which one
is worth routing on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from brandbot import config
from brandbot.gold import spec
from brandbot.intents import taxonomy
from brandbot.llm import client
from brandbot.llm.budget import Budget
from brandbot.pipeline import policy
from brandbot.retrieval.index import Hit, Index

K = 5
REPLY_LIMIT = 280

CLASSIFY_SYSTEM = """You label inbound customer messages for a streaming service's support queue.

Labels:
{labels}

Rules:
{rules}

Label the message on its own. Return JSON only."""

CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": taxonomy.schema_enum(),
        "confidence": {"type": "number"},
    },
    "required": ["intent", "confidence"],
    "additionalProperties": False,
}

DRAFT_SYSTEM = f"""You draft public replies for a streaming service's support account.

You are given past messages and the replies this brand actually sent. Say only what
those past replies support. Do not invent a fix, a timeline, a refund, a link or a
policy that is not in them.

Set answerable to false when the past replies do not answer this message. A wrong
reply costs far more than a handover.

Write at most {REPLY_LIMIT} characters, in the voice of the past replies. Return JSON only."""

DRAFT_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "boolean"},
        "reply": {"type": "string"},
        "used": {"type": "array", "items": {"type": "integer"}},
    },
    "required": ["answerable", "reply", "used"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class Handling:
    thread_id: int
    message: str
    intent: str
    confidence: float
    route: str
    reason: str
    reply: str
    cited: list[int]
    top_score: float
    margin: float
    forced: bool


def _classify(message: str, model: str, budget: Budget | None, replay_only: bool) -> tuple[str, float]:
    system = CLASSIFY_SYSTEM.format(
        labels=taxonomy.prompt_block(), rules="\n".join(spec.CLASSIFIER_RULES)
    )
    result = client.complete(
        model, system, message, schema=CLASSIFY_SCHEMA, budget=budget, replay_only=replay_only
    )
    payload = json.loads(result.text)
    return payload["intent"], float(payload["confidence"])


def _render(hits: list[Hit]) -> str:
    blocks = [
        f"[{i}] customer: {h.precedent.message}\n    brand: {h.precedent.reply}"
        for i, h in enumerate(hits)
    ]
    return "\n\n".join(blocks)


def _draft(
    message: str, hits: list[Hit], model: str, budget: Budget | None, replay_only: bool
) -> tuple[bool, str, list[int]]:
    user = f"Past replies:\n\n{_render(hits)}\n\nNew message:\n{message}"
    result = client.complete(
        model, DRAFT_SYSTEM, user, schema=DRAFT_SCHEMA, budget=budget, replay_only=replay_only
    )
    payload = json.loads(result.text)
    used = [i for i in payload["used"] if 0 <= i < len(hits)]
    return bool(payload["answerable"]), payload["reply"].strip()[:REPLY_LIMIT], used


def handle(
    thread_id: int,
    message: str,
    ix: Index,
    *,
    model: str = config.DRAFT_MODEL,
    classifier: str = config.AGENT_SMALL_MODEL,
    budget: Budget | None = None,
    replay_only: bool = False,
) -> Handling:
    """Classification and drafting run on different models, for a boring reason.

    Groq meters tokens per day per model ID, and one pass over this golden set costs
    about 350k against a 200k ceiling. Splitting the two roles across two IDs is what
    makes the run finish inside the free tier at all.

    Classification gets the smaller model because it is the cheaper task: one label
    from a fixed list, given a taxonomy that spells out its own boundaries.

    Drafting started on the 120b model and moved to qwen when that budget ran out
    part-way through the run. Which model writes the replies is therefore a budget
    outcome, not a considered choice, and the report says so rather than dressing it
    up. Anyone rerunning this with paid capacity should put both roles on one model
    and re-measure before comparing against these numbers.
    """
    hits = ix.search(message, K)
    top = hits[0].score
    margin = top - hits[1].score if len(hits) > 1 else top

    intent, confidence = _classify(message, classifier, budget, replay_only)

    verdict = policy.forced(message, intent, top)
    if verdict is not None:
        return Handling(
            thread_id, message, intent, confidence, spec.ESCALATE, verdict.reason,
            "", [], top, margin, forced=True,
        )

    answerable, reply, used = _draft(message, hits, model, budget, replay_only)
    if not answerable or not reply:
        return Handling(
            thread_id, message, intent, confidence, spec.ESCALATE,
            "the brand's past replies do not answer this message",
            "", [], top, margin, forced=False,
        )

    return Handling(
        thread_id, message, intent, confidence, spec.AUTO, "", reply,
        [hits[i].precedent.thread_id for i in used], top, margin, forced=False,
    )
