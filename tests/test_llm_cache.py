import json

import pytest

from brandbot.llm import cache, client
from brandbot.llm.budget import Budget
from brandbot.llm.cache import Call, Result

CALL = Call("groq", "openai/gpt-oss-120b", "sys", "user", None, 0.0, {})


def test_key_is_stable_across_dict_ordering():
    a = Call("groq", "m", "s", "u", {"a": 1, "b": 2}, 0.0, {"x": 1, "y": 2})
    b = Call("groq", "m", "s", "u", {"b": 2, "a": 1}, 0.0, {"y": 2, "x": 1})
    assert cache.key(a) == cache.key(b)


@pytest.mark.parametrize(
    "field,value",
    [("model", "other"), ("system", "different"), ("user", "different"), ("temperature", 0.7)],
)
def test_changing_any_input_changes_the_key(field, value):
    from dataclasses import replace

    assert cache.key(CALL) != cache.key(replace(CALL, **{field: value}))


def test_roundtrip(tmp_path):
    result = Result("hello", 10, 5)
    cache.put(CALL, result, tmp_path)
    assert cache.get(CALL, tmp_path) == result


def test_entry_stores_the_prompt_that_produced_it(tmp_path):
    cache.put(CALL, Result("hi", 1, 1), tmp_path)
    payload = json.loads(cache.path_for(cache.key(CALL), tmp_path).read_text(encoding="utf-8"))
    assert payload["call"]["user"] == "user"


def test_replay_serves_from_cache_without_a_provider(tmp_path):
    cache.put(CALL, Result("cached", 7, 3), tmp_path)
    budget = Budget()
    got = client.complete(
        CALL.model, CALL.system, CALL.user, budget=budget, replay_only=True, cache_root=tmp_path
    )
    assert got.text == "cached"
    assert budget.spent[CALL.model] == 0  # cached tokens are not spend
    assert budget.cached[CALL.model] == 1


def test_replay_refuses_to_invent_a_missing_response(tmp_path):
    with pytest.raises(client.ReplayMiss):
        client.complete("openai/gpt-oss-120b", "s", "unseen", replay_only=True, cache_root=tmp_path)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("gemini-3.7-flash", "gemini"),
        ("openai/gpt-oss-120b", "groq"),
        ("qwen/qwen3.8-27b", "groq"),
        ("qwen2.5:7b-instruct-q4_K_M", "ollama"),
    ],
)
def test_provider_routing(model, expected):
    assert client._provider(model)[0] == expected
