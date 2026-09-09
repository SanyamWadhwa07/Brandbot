import numpy as np
import pandas as pd
import pytest

from brandbot.data import ingest

HEADER = "tweet_id,author_id,inbound,created_at,text,response_tweet_id,in_response_to_tweet_id\n"
ROWS = (
    "1,sprintcare,False,Tue Oct 31 22:10:47 +0000 2017,@115712 I understand.,2,3\n"
    "2,115712,True,Tue Oct 31 22:11:45 +0000 2017,@sprintcare how,,1\n"
    '3,115712,True,Wed Nov 01 08:00:00 +0000 2017,"@sprintcare line one\nline two",1,\n'
)


@pytest.fixture
def corpus(tmp_path):
    p = tmp_path / "twcs.csv"
    p.write_text(HEADER + ROWS, encoding="utf-8")
    return ingest.build_reply_graph(p)


def test_embedded_newlines_do_not_split_rows(corpus):
    assert len(corpus) == 3


def test_timestamps_land_in_the_right_decade(corpus):
    # A resolution mismatch silently maps every tweet to 1970, which would make the
    # temporal split meaningless without failing anything.
    years = pd.to_datetime(corpus.created_at, unit="s").year
    assert set(years) == {2017}


def test_missing_parent_becomes_sentinel(corpus):
    assert corpus.parent.tolist() == [3, 1, -1]


def test_roundtrip_preserves_every_field(corpus, tmp_path):
    path = tmp_path / "graph.npz"
    ingest.save(corpus, path)
    loaded = ingest.load(path)
    assert np.array_equal(loaded.tweet_id, corpus.tweet_id)
    assert np.array_equal(loaded.created_at, corpus.created_at)
    assert loaded.authors == corpus.authors
