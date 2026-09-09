from brandbot.llm.providers.gemini_provider import to_gemini_schema

SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["a", "b"]},
        "required": {"type": "boolean"},
        "spans": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["intent", "spans"],
    "additionalProperties": False,
    "$schema": "http://json-schema.org/draft-07/schema#",
}


def test_unsupported_keywords_are_dropped():
    out = to_gemini_schema(SCHEMA)
    assert "additionalProperties" not in out
    assert "$schema" not in out


def test_property_names_survive_the_keyword_filter():
    # Filtering field names against the keyword list deletes the fields and leaves
    # "required" referring to properties that are no longer there.
    out = to_gemini_schema(SCHEMA)
    assert set(out["properties"]) == {"intent", "spans", "required"}
    assert out["required"] == ["intent", "spans"]


def test_every_required_field_still_exists():
    out = to_gemini_schema(SCHEMA)
    assert set(out["required"]) <= set(out["properties"])


def test_nested_schemas_are_narrowed_too():
    nested = {"type": "object", "properties": {"x": {"type": "string", "default": "no"}}}
    assert to_gemini_schema(nested)["properties"]["x"] == {"type": "string"}
