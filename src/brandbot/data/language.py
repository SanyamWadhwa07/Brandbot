"""Route messages the agent cannot safely answer to a human.

Measured on this brand's 14,122 conversations: one opening message is genuinely
not English. `langdetect` was tried first and rejected. It labelled "WHERE IS
SEASON 3 OF FOOD WARS" as German and "I'm having major Hulu problems" as
Norwegian, both at 0.9999 confidence, so routing on it would have escalated
plain English and created a failure mode rather than removing one.

What survives is the check that cannot misfire. Latin-script non-English is
below 0.1% here and is left to the escalation rules, which catch a message the
agent cannot ground regardless of the language it is written in.
"""

from __future__ import annotations

import re

# Cyrillic, Arabic, Hebrew, CJK, Hangul, Thai, Devanagari.
NON_LATIN = re.compile(
    r"[Ѐ-ӿ֐-ۿऀ-ॿ฀-๿"
    r"぀-ヿ一-鿿가-힯]"
)
MIN_NON_LATIN_CHARS = 3

# Hashtags and handles carry their own language. The only message this rule flagged
# on real data was English text tagged "#Hulu<japanese>", so they are stripped first.
TAGS = re.compile(r"[#@]\S+")


def needs_translation(text: str) -> bool:
    """True when the message body uses a script the precedents never use."""
    body = TAGS.sub(" ", text)
    return len(NON_LATIN.findall(body)) >= MIN_NON_LATIN_CHARS
