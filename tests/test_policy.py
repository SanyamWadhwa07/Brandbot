from brandbot.pipeline import policy

GROUNDED = 0.9


def fired(message, intent="playback_error", score=GROUNDED):
    verdict = policy.forced(message, intent, score)
    return verdict.reason if verdict else None


def test_a_clean_grounded_message_is_left_to_the_model():
    assert fired("my show keeps buffering on roku") is None


def test_money_outranks_every_other_trigger():
    # The message would also trip the repeat-contact rule; the reason a ticket
    # reaches a human should name the most serious thing about it.
    reason = fired("i already called you and i'm still being charged", intent="billing_refund")
    assert "money or account access" in reason


def test_typographic_apostrophes_do_not_hide_a_trigger():
    assert fired("I’ve tried reinstalling and nothing changed") is not None
    assert fired("I've tried reinstalling and nothing changed") is not None


def test_an_ungrounded_message_escalates_however_ordinary_it_looks():
    assert "nothing to ground a reply in" in fired("my show keeps buffering", score=0.4)


def test_asking_for_a_person_escalates():
    assert "asked for a person" in fired("can i speak to a real human please")
