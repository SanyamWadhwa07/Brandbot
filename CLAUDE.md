# Working conventions for this repo

Read before editing anything here.

## What this project is

A support agent for one Twitter brand, plus the evidence that it works. The brief this was
built for says the proof is worth more than the system, so evaluation integrity outranks
features. When a change would make a number look better without making the system better,
it is the wrong change.

## Scope

Quality over quantity, always. A small number of things that are genuinely finished beats a
long feature list that is broadly half-true. Before building anything, ask what it proves and
who reads it. If the honest answer is "it shows we thought of it", do not build it.

Every feature here has to survive one test: a support lead looking at it asks "would I trust
this in my inbox on Monday?" Nothing gets built to look sophisticated. If a simpler mechanism
answers the same question, the simpler one is correct, and the fact that it is unimpressive is
not an argument against it.

Explicitly out of scope unless the brief demands it: a second brand, multi-turn handling, any
UI, and any abstraction with one call site.

## Attribution

Commits, branch names, tags and PR descriptions carry **no AI attribution**. No
`Co-Authored-By` trailer, no generated-with footer, no mention of an assistant. Every commit
is authored by Sanyam Wadhwa.

AI tooling is disclosed in `docs/REPORT.md` under tooling, which is where the brief's
"cite anything you borrowed" rule actually applies. Disclosure belongs in the report, not
scattered through git metadata.

## Comments

Comments explain **why**, never what. If a reader can see what the line does, the comment is
noise. Write one where a reader would otherwise stop and ask why it was done this way: a
non-obvious ordering constraint, a workaround for a provider bug, a statistical choice with
a defensible alternative.

No docstrings on self-evident functions. No banner comments. No commented-out code. Roughly
one comment per 25-30 lines is the right density here; a file with a comment on every block
has not been thought about.

## Style

Type hints throughout. Dataclasses or pydantic models for structured records, not bare dicts.
Pure functions where practical. No class for logic with one call site. `pathlib`, not
`os.path`. `print` only in the CLI layer; everything else returns values or logs.

Keep modules small and named for what they do. If a module needs a section banner to stay
navigable, it should have been two modules.

## Determinism

Every random draw takes an explicit seed from `config.SEED`. Never call `datetime.now()`
inside anything that gets hashed, cached or committed, because it silently breaks replay.

The response cache is keyed on provider, model and rendered prompt. Changing a prompt
changes the key, which is intended: a cached result must always correspond to the prompt
that produced it.

## The rule that matters most

**Nothing that has seen model output may write to `data/gold/`.**

The golden set and its labels were frozen before any model ran, and the git tags prove it.
Any code path that lets a prediction influence a label destroys the only thing making the
evaluation trustworthy. There is no exception worth making here.

Related: the retrieval index must never contain a thread that appears in the golden set.
`tests/test_no_leakage.py` enforces this and is expected to fail the build, not warn.

## Before committing

```bash
uv run pytest
uv run brandbot eval --replay
```

Both must pass. `--replay` recomputes every headline metric from committed artifacts with no
keys set, which is exactly what a reviewer cloning this repo will do.

## Prose

Documentation in `docs/` is read by humans deciding whether to believe the numbers. Write it
plainly. State what was measured, on what, and what would change the conclusion. No hedging
adjectives, no summary paragraph restating the section above it.

No jargon for its own sake. Technical terms earn their place only when they are the shortest
accurate way to say something, and the first time one appears it gets one plain sentence
saying what it actually asks. "Kappa" is fine once the reader has been told it measures
whether two raters agree more than chance would explain. "Leveraging a multi-faceted
evaluation paradigm" is never fine.

Frame results as support operations, not as machine learning. A reader should see cost per
mishandled ticket and how many messages a human still has to touch, not an abstract score.
