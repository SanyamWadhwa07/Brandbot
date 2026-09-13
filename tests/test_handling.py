from brandbot.intents import handling, taxonomy


def test_every_intent_has_a_group():
    assert set(handling.GROUPS) == set(taxonomy.names())


def test_harmless_and_costly_confusions_are_distinguished():
    assert handling.same_step("app_bug", "service_outage")
    assert not handling.same_step("billing_refund", "playback_error")


def test_money_and_account_never_share_a_group_with_a_fault():
    for intent in ("billing_refund", "account_access"):
        assert handling.group(intent) == handling.HUMAN
        assert not handling.same_step(intent, "playback_error")
