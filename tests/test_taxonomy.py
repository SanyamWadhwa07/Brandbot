from brandbot.intents import taxonomy


def test_names_are_unique():
    assert len(taxonomy.names()) == len(set(taxonomy.names()))


def test_other_is_present_and_last():
    assert taxonomy.names()[-1] == taxonomy.OTHER


def test_size_stays_in_the_range_the_taxonomy_was_designed_for():
    # Too few and routing is useless; too many and a stratified 200 leaves
    # per-class results uninterpretable.
    assert 8 <= len(taxonomy.TAXONOMY) <= 10


def test_every_intent_states_what_it_excludes():
    # An intent defined only by what it includes leaves every boundary case to be
    # re-decided per example, which lowers annotator self-agreement.
    for intent in taxonomy.TAXONOMY:
        assert intent.definition.strip() and intent.excludes.strip()


def test_prompt_block_lists_every_label():
    block = taxonomy.prompt_block()
    for name in taxonomy.names():
        assert name in block


def test_schema_enum_matches_names():
    assert taxonomy.schema_enum()["enum"] == taxonomy.names()
