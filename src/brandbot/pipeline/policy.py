"""Escalation rules that run before the model gets a say.

These implement the triggers in `gold.spec`, which is the same specification the
golden set was labelled against. That shared origin has to be stated plainly
whenever these numbers are quoted: the rule is shared, so the agent is being
scored partly on whether it complies with a specification rather than on whether
it predicted anything.

The detectors are not shared. A human labeller applied the repeat-contact rule by
reading; this applies it by pattern, and the two disagree often enough that the
gap is itself worth measuring. What is not defensible is counting the agreement
as though it were classification skill.

Nothing here consults the model's confidence. A billing dispute goes to a person
whether the draft looked good or not, because the cost of being wrong is not
symmetric and no confidence score prices it correctly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from brandbot.data import language
from brandbot.intents import handling

# A fifth of messages in this corpus use a typographic apostrophe, and "I’ve
# tried" does not match a pattern written with "I've". Normalised at match time
# rather than in the corpus, so the committed slice stays byte-identical while a
# golden set is being labelled against it.
APOSTROPHES = str.maketrans({"‘": "'", "’": "'", "ʼ": "'"})

WANTS_HUMAN = re.compile(
    r"\b(speak|talk) (to|with)\b|\b(real|actual) (person|human)\b|"
    r"\b(supervisor|manager|representative)\b|call me\b|\bphone number\b",
    re.I,
)

# The expensive rule, about one message in ten. "Still", "again" and "every week"
# all say the same thing: the obvious answer has already been tried and failed.
ALREADY_TRIED = re.compile(
    r"\b(still|again|already)\b|\b(second|third|fourth|2nd|3rd|4th) time\b|"
    r"\bevery (day|week|time)\b|\bfor (days|weeks|months)\b|"
    r"\b(i|i've|i have) (tried|restarted|reinstalled|uninstalled|contacted|emailed|called)\b",
    re.I,
)

THREATENS = re.compile(
    r"\b(cancel(l)?ing|cancel my|unsubscrib)\w*\b|\b(lawyer|legal|sue|attorney|bbb)\b|"
    r"\b(switch(ing)? to|going back to) \w+",
    re.I,
)

# Below this the nearest precedent is not about the same problem, so there is
# nothing to ground a reply in.
#
# It is set too low and it is left that way on purpose. Measured afterwards, 99%
# of golden messages clear it, so it escalates almost nothing and contributes
# almost nothing. Similarity does carry real signal about whether a reply will be
# held back: on the retrieval baseline, failures run at 61% below 0.70 and 29%
# above 0.85, and ranking by it beats ranking at random. A floor near 0.78 would
# therefore do actual work.
#
# Moving it there would mean choosing an operating point by looking at the set the
# system is scored on, which is the one shortcut that invalidates the score. The
# badly-placed value is what was committed before the golden set was labelled, so
# it is what runs. Recalibrating it on the dev split is written up as next work.
GROUNDING_FLOOR = 0.62


@dataclass(frozen=True)
class Verdict:
    escalate: bool
    reason: str


def forced(message: str, intent: str, top_score: float) -> Verdict | None:
    """The first trigger that fires, or None when the model's judgement is allowed.

    Ordered by how expensive being wrong is, so the reason a message reaches a
    human names the most serious thing about it rather than whichever pattern
    happened to match first.
    """
    text = message.translate(APOSTROPHES)
    if handling.group(intent) == handling.HUMAN:
        return Verdict(True, f"{intent} touches money or account access, which a person owns")
    if WANTS_HUMAN.search(text):
        return Verdict(True, "the customer asked for a person")
    if THREATENS.search(text):
        return Verdict(True, "the customer is threatening to leave or escalate")
    if ALREADY_TRIED.search(text):
        return Verdict(True, "the customer has already tried a fix or already written in")
    if language.needs_translation(message):
        return Verdict(True, "the message is not in a language the precedents cover")
    if top_score < GROUNDING_FLOOR:
        return Verdict(True, f"nearest precedent scores {top_score:.2f}, nothing to ground a reply in")
    return None
