"""The retrieval index must never be able to see the golden set.

Every other number in the report depends on this one. A precedent drawn from a
golden thread lets the agent retrieve the answer it is being graded on, and no
amount of care elsewhere recovers from that.

The chronological split is what makes this true, so the tests check the property
directly rather than trusting the split code to stay correct.
"""

from __future__ import annotations

import json

import pytest

from brandbot import config
from brandbot.data import brand_slice, splits
from brandbot.gold import sample
from brandbot.retrieval import index


@pytest.fixture(scope="module")
def parts():
    return splits.partition(brand_slice.load())


@pytest.fixture(scope="module")
def precedent_ids(parts):
    return {
        p.thread_id
        for p in map(index.to_precedent, parts[splits.Split.INDEX])
        if p is not None
    }


def test_splits_share_no_thread(parts):
    ids = {split: {c.thread_id for c in convs} for split, convs in parts.items()}
    assert not ids[splits.Split.INDEX] & ids[splits.Split.GOLD]
    assert not ids[splits.Split.INDEX] & ids[splits.Split.DEV]
    assert not ids[splits.Split.DEV] & ids[splits.Split.GOLD]


def test_every_precedent_predates_every_golden_thread(parts):
    latest_index = max(c.started_at for c in parts[splits.Split.INDEX])
    earliest_gold = min(c.started_at for c in parts[splits.Split.GOLD])
    assert latest_index < earliest_gold


def test_no_precedent_comes_from_a_golden_thread(parts, precedent_ids):
    gold = {c.thread_id for c in parts[splits.Split.GOLD]}
    assert not precedent_ids & gold


def test_no_sampled_candidate_is_a_precedent(precedent_ids):
    if not sample.CANDIDATES_PATH.exists():
        pytest.skip("no sampling record in this checkout")
    with sample.CANDIDATES_PATH.open(encoding="utf-8") as handle:
        drawn = {json.loads(line)["thread_id"] for line in handle}
    assert drawn and not drawn & precedent_ids


def test_no_labelled_thread_is_a_precedent(precedent_ids):
    labelled = config.GOLD / "golden.jsonl"
    if not labelled.exists():
        pytest.skip("golden set not labelled yet")
    with labelled.open(encoding="utf-8") as handle:
        ids = {json.loads(line)["thread_id"] for line in handle}
    assert ids and not ids & precedent_ids


def test_golden_file_carries_no_model_derived_field():
    """`stratum` is chosen by an embedding model, so it lives in the sampling
    record and is joined at load time. Finding it inside the labelled file would
    mean something model-derived had been written into `data/gold/`."""
    labelled = config.GOLD / "golden.jsonl"
    if not labelled.exists():
        pytest.skip("golden set not labelled yet")
    with labelled.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    forbidden = {"stratum", "predicted", "confidence", "intent_pred", "score"}
    assert rows and not any(forbidden & set(row) for row in rows)
