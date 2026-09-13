"""The floor: guess the commonest intent, send the same line to everyone, never escalate.

This exists to price the other systems. A macro-F1 of 0.4 means nothing until you
know that answering every message identically scores 0.05, and an auto-send rate
of 70% means nothing until you know what sending to everyone costs.

Never escalating is the point rather than an oversight. It is the cheapest possible
policy and it is what a system with no notion of its own limits would do, so it sets
the worst-case risk the escalation work has to beat.

The canned reply is hand-written, not mined. The brand almost never repeats itself:
9,822 distinct replies across 9,886 precedents, and its most frequent line is an
outage notice from one particular night. This is the brand's modal move (apologise,
ask which device) written out once.
"""

from __future__ import annotations

from brandbot.gold import spec
from brandbot.pipeline.agent import Handling

# Modal intent on the dev split under the frozen clustering: 660 of 2,542, 26%.
MAJORITY_INTENT = "content_availability"

CANNED = "Oh no! Sorry for the trouble. Which device are you watching on?"


def handle(thread_id: int, message: str) -> Handling:
    return Handling(
        thread_id=thread_id,
        message=message,
        intent=MAJORITY_INTENT,
        confidence=1.0,
        route=spec.AUTO,
        reason="",
        reply=CANNED,
        cited=[],
        top_score=0.0,
        margin=0.0,
        forced=False,
    )
