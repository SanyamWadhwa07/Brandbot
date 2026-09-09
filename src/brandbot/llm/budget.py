"""Track token spend per model.

Groq's free tier meters requests, tokens per minute and tokens per day
separately, and enforces them per model ID rather than per account. A run that
blows the daily cap cannot resume until tomorrow, so spend is worth watching
before it is spent rather than after.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

# Free-tier daily token ceilings, per model ID.
DAILY_TOKENS = {
    "openai/gpt-oss-120b": 200_000,
    "openai/gpt-oss-20b": 200_000,
    "qwen/qwen3.8-27b": 200_000,
    "qwen/qwen3.6-27b": 200_000,
}


@dataclass
class Budget:
    spent: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    cached: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, model: str, tokens: int, was_cached: bool) -> None:
        self.calls[model] += 1
        if was_cached:
            self.cached[model] += 1
        else:
            self.spent[model] += tokens

    def remaining(self, model: str) -> int | None:
        cap = DAILY_TOKENS.get(model)
        return None if cap is None else cap - self.spent[model]

    def report(self) -> list[str]:
        lines = []
        for model in sorted(set(self.calls)):
            cap = DAILY_TOKENS.get(model)
            budget = f" of {cap:,} daily" if cap else ""
            lines.append(
                f"{model}: {self.calls[model]:,} calls "
                f"({self.cached[model]:,} cached), {self.spent[model]:,} tokens{budget}"
            )
        return lines
