from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from brandbot import config

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
