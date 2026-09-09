"""The frozen intent taxonomy for hulu_support.

Induced by over-clustering the dev split into 20 groups, naming each one, then
merging by hand. Frozen and tagged before any golden-set label was written and
before any agent output existed, so the label set cannot have been shaped by what
the model turned out to be good at. The tag `v0.1-taxonomy-frozen` is the record.

Each definition says what belongs *and* what does not. Boundary cases are where
annotator disagreement comes from, and an intent defined only by what it includes
leaves every boundary to be re-decided per example, which shows up later as a
lower self-agreement ceiling.

Merged from clusters (dev counts in brackets):
    playback_error        <- 10 [122], 19 [81], 1 [82], 9 [50], 15 [147]
    content_availability  <- 17 [182], 7 [175], 4 [173], 14 [130]
    app_bug               <- 6 [122], 0 [95]
    live_tv_channels      <- 5 [107], 18 [91]
    ads_complaint         <- 8 [167]
    billing_refund        <- 3 [134]
    watchlist_issue       <- 2 [121]
    account_access        <- 16 [110]
    service_outage        <- part of 12 [248]
    other                 <- 11 [163], 13 [42], remainder of 12
"""

from __future__ import annotations

from dataclasses import dataclass

OTHER = "other"


@dataclass(frozen=True)
class Intent:
    name: str
    definition: str
    excludes: str


TAXONOMY: tuple[Intent, ...] = (
    Intent(
        "playback_error",
        "Content that should play fails to start, stalls, buffers or throws an error.",
        "a channel missing from the lineup, or the app failing before playback begins.",
    ),
    Intent(
        "content_availability",
        "Asking whether a title, season or episode is on the service, when it arrives, "
        "or asking for it to be added.",
        "content that exists on the service but will not play.",
    ),
    Intent(
        "app_bug",
        "The app crashes, freezes, navigates to the wrong place or has broken controls "
        "on a specific device.",
        "content that plays but stutters, which is a playback error.",
    ),
    Intent(
        "live_tv_channels",
        "A live TV channel is missing from the lineup, unavailable in a region, or not "
        "offered on a device.",
        "a listed live channel that errors during playback.",
    ),
    Intent(
        "ads_complaint",
        "Too many ads, the same ad repeating, or a request for ad-free viewing on a paid plan.",
        "an ad break that breaks playback, which is a playback error.",
    ),
    Intent(
        "billing_refund",
        "A charge is disputed, a refund is wanted, or billing continued after cancelling.",
        "questions about plan pricing or features where nothing is being disputed.",
    ),
    Intent(
        "watchlist_issue",
        "The watchlist or My Stuff does not show new episodes, or does not update.",
        "shows that are genuinely absent from the catalogue.",
    ),
    Intent(
        "account_access",
        "Cannot sign in: password or email recovery, verification failures, locked account.",
        "a signed-in customer disputing a charge.",
    ),
    Intent(
        "service_outage",
        "Reporting or asking about a failure that appears to affect many customers at once.",
        "a fault on one device or with one title.",
    ),
)


def names() -> list[str]:
    return [i.name for i in TAXONOMY] + [OTHER]


def prompt_block() -> str:
    """Compact rendering for prompts.

    Carried by every classification call, so it is deliberately terse: on a
    metered free tier a verbose taxonomy is paid for on every single message.
    """
    lines = [f"{i.name}: {i.definition} Not: {i.excludes}" for i in TAXONOMY]
    lines.append(f"{OTHER}: none of the above fits.")
    return "\n".join(lines)


def schema_enum() -> dict:
    return {"type": "string", "enum": names()}
