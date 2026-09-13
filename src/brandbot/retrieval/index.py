"""Find how this brand answered similar messages before.

Precedents come only from the index split, which is the earliest 70% of the
corpus. Retrieval in deployment always looks backwards, and a precedent written
after the message it is answering is a result that could never hold in production.
The chronological split makes that structural rather than a rule someone has to
remember: the golden set is the latest slice, so it cannot appear here.

A precedent is the customer's opening message paired with the brand's first
public reply. On this brand that pairing is the whole interaction 63% of the time
and the median conversation has exactly one brand turn, so the first reply is the
response rather than a holding message.

Threads another company also replied in are dropped. The corpus does not record
which company wrote an outbound tweet, so a customer who tagged several support
accounts leaves another team's words looking like this brand's. Left in, they get
retrieved and copied: one draft in an early sample answered a buffering complaint
by asking for the customer's street address and signing off as another carrier.

`resolved` records whether the customer's last message reads as a positive
sign-off. It is a weak signal and it is stored rather than filtered on: only 17%
of threads end that way, so filtering would throw away most of the corpus to chase
a label that is itself a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from brandbot.data import brand_slice
from brandbot.data.brand_slice import Conversation
from brandbot.retrieval import embed

SIGN_OFF = re.compile(
    r"\b(thank|thanks|thx|ty|fixed|worked|working now|resolved|got it|appreciate|perfect)\b",
    re.I,
)


@dataclass(frozen=True)
class Precedent:
    thread_id: int
    message: str
    reply: str
    resolved: bool


@dataclass(frozen=True)
class Hit:
    precedent: Precedent
    score: float


@dataclass(frozen=True)
class Index:
    precedents: list[Precedent]
    vectors: np.ndarray

    def search(self, message: str, k: int = 5) -> list[Hit]:
        query = embed.embed([message], embed.QUERY)[0]
        scores = self.vectors @ query
        # argpartition finds the top k without ordering the other 9,000.
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        return [
            Hit(self.precedents[i], float(scores[i]))
            for i in top[np.argsort(-scores[top])]
        ]


def to_precedent(conversation: Conversation) -> Precedent | None:
    reply = next((t["text"] for t in conversation.turns if t["role"] == "brand"), None)
    if not reply:
        return None
    customer = [t["text"] for t in conversation.turns if t["role"] == "customer"]
    return Precedent(
        thread_id=conversation.thread_id,
        message=conversation.opening["text"],
        reply=reply,
        resolved=bool(SIGN_OFF.search(customer[-1])) if len(customer) > 1 else False,
    )


def build(conversations: list[Conversation]) -> Index:
    foreign = brand_slice.load_foreign()
    precedents = [
        p
        for p in map(to_precedent, conversations)
        if p is not None and p.thread_id not in foreign
    ]
    return Index(precedents, embed.embed([p.message for p in precedents], embed.DOCUMENT))
