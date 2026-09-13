from brandbot.data.brand_slice import Conversation
from brandbot.retrieval import index


def conversation(thread_id, *turns):
    return Conversation(
        thread_id=thread_id,
        brand="hulu_support",
        started_at=0,
        turns=[{"tweet_id": thread_id + i, "role": r, "created_at": i, "text": t}
               for i, (r, t) in enumerate(turns)],
    )


def test_a_thread_the_brand_never_answered_is_not_a_precedent():
    assert index.to_precedent(conversation(1, ("customer", "anyone there?"))) is None


def test_the_precedent_is_the_first_brand_reply():
    p = index.to_precedent(
        conversation(2, ("customer", "help"), ("brand", "first"), ("brand", "second"))
    )
    assert p.reply == "first"


def test_resolution_needs_a_customer_turn_after_the_reply():
    # A sign-off in the opening message says nothing about whether the reply worked.
    opener_only = index.to_precedent(conversation(3, ("customer", "thanks anyway"), ("brand", "hi")))
    assert not opener_only.resolved

    answered = index.to_precedent(
        conversation(4, ("customer", "help"), ("brand", "try this"), ("customer", "that fixed it"))
    )
    assert answered.resolved


def test_threads_another_company_replied_in_are_excluded():
    from brandbot.data import brand_slice

    foreign = brand_slice.load_foreign()
    assert foreign, "the exclusion list should not be empty"
    convs = [conversation(t, ("customer", "help"), ("brand", "hi")) for t in list(foreign)[:3]]
    assert index.build(convs).precedents == []
