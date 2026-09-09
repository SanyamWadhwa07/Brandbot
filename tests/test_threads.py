import numpy as np
import pytest

from brandbot.data.ingest import ReplyGraph
from brandbot.data import threads


def graph(rows):
    """rows: (tweet_id, author, inbound, created_at, parent)"""
    tid, auth, inb, ts, par = zip(*rows, strict=True)
    names = sorted({a for a in auth})
    return ReplyGraph(
        tweet_id=np.array(tid, dtype=np.int64),
        author=np.array([names.index(a) for a in auth], dtype=np.int32),
        authors=names,
        inbound=np.array(inb, dtype=bool),
        created_at=np.array(ts, dtype=np.int64),
        parent=np.array(par, dtype=np.int64),
    )


def test_chain_collapses_to_one_root():
    g = graph([
        (10, "cust", True, 100, -1),
        (11, "brand", False, 200, 10),
        (12, "cust", True, 300, 11),
    ])
    roots = threads.resolve_roots(g)
    assert len(set(roots.tolist())) == 1


def test_parent_outside_the_corpus_is_treated_as_a_root():
    # in_response_to_tweet_id often points at a tweet the dump does not contain.
    g = graph([(10, "cust", True, 100, 999)])
    assert threads.resolve_roots(g).tolist() == [0]


def test_hub_with_several_customers_is_not_a_conversation():
    g = graph([
        (1, "brand", False, 100, -1),
        (2, "custA", True, 200, 1),
        (3, "custB", True, 300, 1),
    ])
    roots = threads.resolve_roots(g)
    t = threads.build_threads(g, roots)
    assert t.n_customers.tolist() == [2]
    assert not t.is_conversation().any()


def test_single_customer_conversation_is_kept():
    g = graph([
        (1, "cust", True, 100, -1),
        (2, "brand", False, 200, 1),
        (3, "cust", True, 300, 2),
    ])
    roots = threads.resolve_roots(g)
    t = threads.build_threads(g, roots)
    assert t.is_conversation().tolist() == [True]
    assert t.size.tolist() == [3]


def test_turns_come_back_in_time_order():
    g = graph([
        (1, "cust", True, 300, -1),
        (2, "brand", False, 100, 1),
        (3, "cust", True, 200, 2),
    ])
    roots = threads.resolve_roots(g)
    turns = threads.turns_by_thread(roots, g)
    (rows,) = turns.values()
    assert g.created_at[rows].tolist() == [100, 200, 300]
