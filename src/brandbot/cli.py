from __future__ import annotations

import json
import sys
from collections import Counter

import typer
from rich.console import Console
from rich.table import Table

from brandbot import config
from brandbot.gold import labeller, sample

app = typer.Typer(add_completion=False, help="Grounded support agent for one Twitter brand.")
console = Console()


@app.callback()
def main() -> None:
    """Keeps subcommand names stable while the command set is still small."""


@app.command()
def doctor() -> None:
    """Report whether this checkout can run, and which paths are the live ones."""
    table = Table(show_header=False, box=None)
    table.add_row("python", sys.version.split()[0])
    table.add_row("root", str(config.ROOT))
    table.add_row("GROQ_API_KEY", "set" if config.groq_key() else "unset")
    table.add_row("GEMINI_API_KEY", "set" if config.gemini_key() else "unset")

    for label, path in [
        ("raw corpus", config.RAW),
        ("brand slice", config.BRAND),
        ("golden set", config.GOLD),
        ("artifacts", config.ARTIFACTS),
    ]:
        files = sorted(p.name for p in path.glob("*") if p.is_file()) if path.exists() else []
        table.add_row(label, ", ".join(files) if files else "[dim]empty[/dim]")

    console.print(table)
    console.print("\n[dim]Keys are only needed for live runs. `eval --replay` needs none.[/dim]")


@app.command()
def label(
    review: bool = typer.Option(
        False, help="pre-fill each example with the reference label and review it"
    ),
) -> None:
    """Sample golden-set candidates and write the offline labelling page."""
    candidates = sample.build(sample.CANDIDATES_PATH)
    prefill = None
    name = "golden-v1"
    if review:
        reference = config.DATA / "interim" / "reference_labels.jsonl"
        prefill = {
            str(row["id"]): row
            for row in map(json.loads, reference.read_text(encoding="utf-8").splitlines())
        }
        name = "golden-v1-review"
    page = labeller.render(candidates, config.ARTIFACTS / "label.html", name, prefill)

    counts = Counter(c.stratum for c in candidates)
    console.print(f"{len(candidates)} candidates: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    console.print(f"sampling record  {sample.CANDIDATES_PATH}")
    console.print(f"labelling page   {page}")
    console.print(f"\n[dim]Label every example, then Download JSONL into {config.GOLD}/golden.jsonl[/dim]")


@app.command()
def run(
    system: str = typer.Argument(..., help="agent, nearest or majority"),
    limit: int = typer.Option(0, help="stop after N messages, for a cheap dry run"),
) -> None:
    """Run one system over the golden messages and record what it did."""
    from brandbot.eval import run as runner
    from brandbot.llm.budget import Budget

    rows = [json.loads(line) for line in sample.CANDIDATES_PATH.read_text(encoding="utf-8").splitlines()]
    messages = [(r["thread_id"], r["text"]) for r in rows][: limit or None]

    budget = Budget()
    result = runner.execute(system, messages, budget=budget)
    path = runner.save(result)

    auto = sum(1 for h in result.handlings if h.route == "auto")
    sent = sum(1 for v in result.verdicts if v.send)
    console.print(f"{system}: {len(result.handlings)} messages, {auto} auto-sent, {len(result.verdicts)} judged")
    console.print(f"judge would send {sent}/{len(result.verdicts)}" if result.verdicts else "nothing judged")
    for line in budget.report():
        console.print(f"[dim]{line}[/dim]")
    console.print(f"wrote {path}")


@app.command(name="eval")
def evaluate(
    replay: bool = typer.Option(True, help="recompute from committed runs, never call a model"),
    cost_ratio: float = typer.Option(5.0, help="human touches a wrongly auto-sent reply costs"),
) -> None:
    """Recompute every headline number from the committed run records."""
    from brandbot.eval import report
    from brandbot.eval import run as runner
    from brandbot.gold import store

    if not store.GOLDEN_PATH.exists():
        console.print("[red]No golden set yet.[/red] Label it first: `uv run brandbot label`")
        raise typer.Exit(1)

    labelled = store.load()
    counts = store.counts(labelled)
    console.print(
        f"Golden set: {len(labelled)} messages "
        f"({counts['random']} random, {counts['enriched']} enriched), "
        f"{counts['revealed']} revealed the thread, {counts['multi_intent']} multi-intent"
    )

    systems = [s for s in (runner.AGENT, runner.NEAREST, runner.MAJORITY)
               if (runner.RUNS / f"{s}.json").exists()]
    if not systems:
        console.print("[red]No runs recorded.[/red] Run one first: `uv run brandbot run agent`")
        raise typer.Exit(1)

    runs = {s: runner.load(s) for s in systems}
    cards = {s: report.score(runs[s], labelled, cost_ratio=cost_ratio) for s in systems}

    table = Table(box=None, pad_edge=False)
    for col in ("system", "n", "auto-sent", "of those, held", "human touches/100",
                "macro-F1", "same next step", "prior-weighted"):
        table.add_column(col, justify="right" if col != "system" else "left")
    for s in systems:
        c = cards[s]
        table.add_row(
            s, str(c.n), f"{c.coverage:.0%}", f"{c.risk:.0%}", f"{c.touches_per_100:.0f}",
            str(c.macro_f1), str(c.same_step), f"{c.prior_weighted:.0%}",
        )
    console.print()
    console.print(table)
    console.print(
        "\n[dim]auto-sent is coverage; held is the share of sent replies a support lead would stop.\n"
        "Neither means anything alone: escalate everything and held goes to zero.[/dim]"
    )

    if runner.AGENT in cards:
        console.print()
        for other in (runner.NEAREST, runner.MAJORITY):
            if other not in runs:
                continue
            for metric, comp in report.against(runs[runner.AGENT], runs[other], labelled).items():
                verdict = "clears noise" if comp.significant else "[yellow]inside noise[/yellow]"
                console.print(f"agent vs {other}, {metric}: {comp.delta} {verdict}")

    console.print()
    for s in systems:
        worst = cards[s].intent.worst(3)
        console.print(f"[dim]{s} weakest intents: " +
                      ", ".join(f"{w.label} F1 {w.f1:.2f} (n={w.support})" for w in worst) + "[/dim]")


@app.command()
def rate(per_system: int = typer.Option(27, help="replies to draw from each system")) -> None:
    """Write the blind reply-review page used to check the judge against a human."""
    from brandbot.eval import rater
    from brandbot.eval import run as runner

    systems = [s for s in (runner.AGENT, runner.NEAREST, runner.MAJORITY)
               if (runner.RUNS / f"{s}.json").exists()]
    if not systems:
        console.print("[red]No runs recorded.[/red] Run one first: `uv run brandbot run agent`")
        raise typer.Exit(1)

    sampled = rater.draw([runner.load(s) for s in systems], per_system=per_system)
    page = rater.render(sampled, config.ARTIFACTS / "rate.html", "judge-check-v1")
    key = rater.key(sampled, config.DATA / "interim" / "rating_key.jsonl")

    console.print(f"{len(sampled)} replies drawn from {', '.join(systems)}")
    console.print(f"review page  {page}")
    console.print(f"answer key   {key}  [dim](which system wrote what; not shown on the page)[/dim]")
    console.print(f"\n[dim]Rate every reply, then Download JSONL into {config.GOLD}/ratings.jsonl[/dim]")
