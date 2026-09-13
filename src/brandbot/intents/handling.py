"""What a support team actually does with a message, as opposed to what it is about.

The taxonomy labels the topic of the complaint. A support lead does not staff by
topic, they staff by next step, and on this brand the two come apart badly. Measured
on the dev split, the next action Hulu took after an `app_bug` and after a
`service_outage` is 96% the same distribution; `live_tv_channels` and `other` are
95% the same. Only three intents have a move of their own: `playback_error` asks
which device, `content_availability` cites licensing, `watchlist_issue`
acknowledges a known issue.

That is why macro-F1 alone is the wrong headline. It charges the same price for
confusing `app_bug` with `service_outage`, which changes nothing, as for confusing
`billing_refund` with `playback_error`, which auto-replies to a money dispute. The
group a prediction lands in is what decides whether the mistake costs anything.

Reported next to accuracy, never instead of it. Grouping is coarser and will always
look better, and quoting only the flattering number is the failure this project
exists to avoid.
"""

from __future__ import annotations

from brandbot.intents.taxonomy import OTHER

TROUBLESHOOT = "troubleshoot"
CATALOGUE = "catalogue"
POLICY = "policy"
HUMAN = "human"
ACKNOWLEDGE = "acknowledge"

GROUPS: dict[str, str] = {
    "playback_error": TROUBLESHOOT,
    "app_bug": TROUBLESHOOT,
    "live_tv_channels": TROUBLESHOOT,
    "service_outage": TROUBLESHOOT,
    "watchlist_issue": TROUBLESHOOT,
    "content_availability": CATALOGUE,
    "ads_complaint": POLICY,
    "billing_refund": HUMAN,
    "account_access": HUMAN,
    OTHER: ACKNOWLEDGE,
}


def group(intent: str) -> str:
    return GROUPS[intent]


def same_step(truth: str, predicted: str) -> bool:
    """Whether the mistake, if it is one, changes what happens to the customer."""
    return GROUPS[truth] == GROUPS[predicted]
