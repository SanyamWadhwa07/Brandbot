"""Machine-readable form of the hand merge recorded in `taxonomy`.

The merge itself lives in that module's docstring and was frozen at
`v0.1-taxonomy-frozen`. It is transcribed here rather than added there so the
tagged file stays byte-identical to what the freeze covers.

Used only to point sampling at the thin intents. Cluster 12 is genuinely mixed,
holding both outage reports and unrelated leftovers, and is mapped whole to
`service_outage`: for choosing which messages to *show* a labeller, a cluster
that is half outages is a far better lead than a uniform draw, and the label the
labeller writes settles what it actually was.
"""

from __future__ import annotations

from brandbot.intents.taxonomy import OTHER

CLUSTER_MERGE: dict[int, str] = {
    10: "playback_error", 19: "playback_error", 1: "playback_error",
    9: "playback_error", 15: "playback_error",
    17: "content_availability", 7: "content_availability",
    4: "content_availability", 14: "content_availability",
    6: "app_bug", 0: "app_bug",
    5: "live_tv_channels", 18: "live_tv_channels",
    8: "ads_complaint",
    3: "billing_refund",
    2: "watchlist_issue",
    16: "account_access",
    12: "service_outage",
    11: OTHER, 13: OTHER,
}
