"""The annotation spec the golden set is labelled against.

Written before any agent existed, from decisions made on real examples pulled out
of the candidate pool, so the agent is measured against a standard it did not get
to shape. Every rule here settles a case that came up in the data, not a case
someone imagined might.

The route label deliberately is not a function of the intent. If "billing_refund
always escalates" were the whole rule, route accuracy would be intent accuracy
wearing a different name and the escalation numbers would measure nothing.

What the brand actually did is not the ground truth. Hulu's routing was driven by
who was staffed at 2am; these labels are about what should happen.
"""

from __future__ import annotations

from dataclasses import dataclass

AUTO = "auto"
ESCALATE = "escalate"


@dataclass(frozen=True)
class Route:
    code: str
    key: str
    definition: str


ROUTES: tuple[Route, ...] = (
    Route(
        AUTO,
        "A",
        "A reply built from the brand's past replies would genuinely move this "
        "forward, needs no account access, and promises nothing only a human can "
        "deliver. Asking the customer a clarifying question counts as auto.",
    ),
    Route(
        ESCALATE,
        "E",
        "A human is needed: the fix is account-specific, money is in dispute, the "
        "customer asks for a person, the customer has tried or written before, or "
        "the brand's history holds no answer to give.",
    ),
)

# Conditions that force escalation whatever the model's confidence.
#
# The repeat-contact rule is the expensive one. It moves about one message in ten
# straight to a human, and it was chosen over a narrower "only if they say they
# contacted support" reading because the narrow version could not be applied the
# same way twice: "I restarted my router" and "I tweeted you last week" shade into
# each other, and a rule a human cannot follow consistently is not a rule.
#
# This has a consequence the report has to state plainly. These triggers are
# detected by the same deterministic code the labeller applied by hand, so the
# agent scores near-perfectly on the messages they catch. That is compliance with
# a specification, not prediction, and counting it as accuracy would flatter the
# system. The routing evaluation that means anything runs on everything else.
TRIGGERS = (
    "money is disputed, refunded, or charged after a cancellation",
    "the fix requires acting on this specific account",
    "the customer asks for a human, a manager, or a callback",
    "the customer says they already tried a fix, or already wrote in",
    "the message threatens cancellation, legal action, or public escalation",
    "nothing in the brand's history answers the question",
)

# How to choose one intent when a message carries more than one.
#
# Rule 1 came from "I thought I had canceled but was still charged $30". Two
# intents, and the premise is unverifiable: nobody can tell from the tweet whether
# the cancellation went through. The label follows the outcome the customer wants,
# because that is what a support queue would route on. The unverifiable premise is
# a drafting problem, not a labelling one, and it is handled by letting the reply
# ask rather than assert.
#
# Rule 2 keeps `other` meaningful. A complaint with no recoverable request is not
# a weak example of some intent, it is a different kind of message, and the right
# reply to it is an acknowledgement rather than a fix.
# A second, orthogonal label: what the customer wants done, as opposed to what the
# message is about.
#
# Added during labelling, after the frozen taxonomy turned out to capture the wrong
# axis. Clustering groups messages by subject, because subject is what the words
# carry, but what decides whether a bot can handle a message is the kind of thing
# being asked for. On this pool 57% of messages are a question, and the brand's
# habitual answer to a question is a diagnostic question back, which is why one
# canned line clears the judge on 54% of messages.
#
# The taxonomy is not being replaced. It stays frozen and tagged so the claim that
# the label set predates any model output still holds. This is recorded alongside
# it and used for analysis, not fed to a classifier.
INFORMATION = "information"
ACTION = "action"
ACKNOWLEDGEMENT = "acknowledgement"

ASK_TYPES: tuple[Route, ...] = (
    # The test is one question: could someone with no access to this customer's
    # account satisfy them? A troubleshooting step is information even though the
    # customer wants their problem gone, because telling them is enough. A refund
    # is action because no amount of explaining produces it.
    #
    # Stated this way after the first labelling batch, where `action` was being
    # read as "wants a fix" rather than "needs their account touched", and 22 of 52
    # messages landed there. The earlier wording invited it.
    Route(
        INFORMATION,
        "J",
        "A good answer would satisfy them, with no access to their account: why it "
        "happens, when it arrives, whether it exists, what to try next.",
    ),
    Route(
        ACTION,
        "K",
        "Somebody must change something on their account or systems before anything "
        "improves: a refund, an unlock, a credit, a plan change. No answer is enough.",
    ),
    Route(
        ACKNOWLEDGEMENT,
        "L",
        "There is no request in it. They want to be heard, not helped.",
    ),
)

# Conventions a model classifying one message can actually act on. These go in the
# classifier's prompt so it labels by the same convention the golden set was
# labelled by, which is specification rather than leakage: it is told the rule, not
# the answer.
CLASSIFIER_RULES = (
    "When a message carries two intents, label the outcome the customer wants, "
    "not the problem that caused it.",
    "A complaint with no request in it is `other`, whatever it is about.",
    "Judge the message on its own. Do not assume context it does not contain.",
)

# Everything the human labeller works from. The last two have no meaning to a model
# looking at one message: it has no thread to reveal and writes no ask-type. Sending
# them to the classifier would be instructions it cannot follow, so the prompt takes
# `CLASSIFIER_RULES` instead.
TIE_BREAKS = (
    *CLASSIFIER_RULES[:2],
    "Ask-type is what would satisfy them, not how likely they are to get it.",
    "Label the opening message on its own. Reveal the thread only when genuinely "
    "stuck, and accept that revealing is recorded.",
)
