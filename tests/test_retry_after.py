"""The wait a provider asks for is the wait to take.

Exponential backoff into a per-minute token bucket is the worst of both: the early
retries are too short to clear the window and the later ones sleep far past it.
"""

from __future__ import annotations

import pytest

from brandbot.llm import client


@pytest.mark.parametrize(
    "message, expected",
    [
        ("Rate limit reached. Please try again in 6.394s.", 6.394),
        ("Please try again in 1m13.2s", 73.2),
        ("Please try again in 0.5s", 0.5),
    ],
)
def test_reads_the_wait_the_provider_asked_for(message, expected):
    assert client._requested_wait(RuntimeError(message)) == pytest.approx(expected)


def test_falls_back_when_no_wait_is_offered():
    assert client._requested_wait(RuntimeError("503 service unavailable")) is None
