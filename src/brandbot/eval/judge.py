"""Grade a drafted reply the way a support lead reviewing a queue would.

Four questions, each yes or no, then one holistic call: would you let this send
unedited? Binary rather than a 1-5 scale on purpose. Nobody can say what the
difference between a 3 and a 4 is, two raters will not draw it in the same place,
and a scale that cannot be applied consistently produces an agreement statistic
that flatters itself. "Would you send it" is a decision a support lead makes
dozens of times a day.

The judge sees the message, the reply and the precedents that were retrieved. It
never sees which system wrote the reply, so the agent and the baselines are graded
under identical conditions.

The judge is Gemini and every system it grades is a different family, which matters
because an LLM grading its own family rewards replies for sounding like something
it would have written. That removes the worst of the effect rather than all of it,
so the report re-grades a sample with a third model and states the gap.

The holistic `send` is the headline. The four dimensions exist to say *why*
something failed, and they are not summed into it: where `send` disagrees with all
four dimensions passing, the disagreement is reported rather than resolved, because
it is the cheapest available evidence about how steady the judge is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from brandbot.llm import client
from brandbot.llm.budget import Budget
from brandbot.retrieval.index import Hit

SYSTEM = """You review draft replies before they are sent from a streaming service's
public support account. You are given the customer's message, the replies this brand
actually sent to similar messages in the past, and one draft reply.

Judge the draft against the past replies, not against what you would have written.

grounded: every fact, fix, link, timeline and policy in the draft is supported by the
past replies. A plausible invention is not grounded.
addresses: it answers what this customer actually asked, not a neighbouring question.
safe: it promises nothing the brand would be held to and that the past replies do not
already promise. Refunds, dates, and guarantees are the usual offenders.
tone: it reads like the brand's own support account.

send: would you let this go out unedited, to this customer, right now? Asking the
customer a clarifying question is a legitimate reply and can pass.

Give a reason of at most 20 words naming the single most important problem, or why it
is fine. Return JSON only."""

SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "addresses": {"type": "boolean"},
        "safe": {"type": "boolean"},
        "tone": {"type": "boolean"},
        "send": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["grounded", "addresses", "safe", "tone", "send", "reason"],
    "additionalProperties": False,
}

DIMENSIONS = ("grounded", "addresses", "safe", "tone")


@dataclass(frozen=True)
class Verdict:
    thread_id: int
    grounded: bool
    addresses: bool
    safe: bool
    tone: bool
    send: bool
    reason: str

    @property
    def all_dimensions_pass(self) -> bool:
        return all(getattr(self, d) for d in DIMENSIONS)

    @property
    def failed(self) -> list[str]:
        return [d for d in DIMENSIONS if not getattr(self, d)]


def render(message: str, reply: str, hits: list[Hit]) -> str:
    past = "\n\n".join(
        f"customer: {h.precedent.message}\n    brand: {h.precedent.reply}" for h in hits
    )
    return f"Past replies:\n\n{past}\n\nCustomer message:\n{message}\n\nDraft reply:\n{reply}"


def grade(
    thread_id: int,
    message: str,
    reply: str,
    hits: list[Hit],
    *,
    model: str | None = None,
    budget: Budget | None = None,
    replay_only: bool = False,
) -> Verdict:
    user = render(message, reply, hits)
    result = (
        client.judge(user, SYSTEM, SCHEMA, budget=budget, replay_only=replay_only)
        if model is None
        else client.complete(
            model, SYSTEM, user, schema=SCHEMA, budget=budget, replay_only=replay_only
        )
    )
    payload = json.loads(result.text)
    return Verdict(
        thread_id=thread_id,
        grounded=bool(payload["grounded"]),
        addresses=bool(payload["addresses"]),
        safe=bool(payload["safe"]),
        tone=bool(payload["tone"]),
        send=bool(payload["send"]),
        reason=payload["reason"].strip(),
    )
