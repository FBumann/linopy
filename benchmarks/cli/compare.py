"""Snapshot comparison command: ``compare``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from benchmarks.cli._base import _suggest_snapshots, app


@app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def compare(ctx: typer.Context) -> None:
    """
    Compare timing snapshots side-by-side via ``pytest-benchmark compare``.

    Thin wrapper around the upstream tool so the whole suite stays under
    one entry point. Pass the snapshot paths first, then any pytest-benchmark
    flags::

        python -m benchmarks compare a.json b.json
        python -m benchmarks compare a.json b.json --group-by=name
        python -m benchmarks compare a.json b.json --histogram=plots/cmp

    With no arguments (or missing paths), prints what snapshots exist
    under ``.benchmarks/`` so you can copy-paste the path you want.

    Memory snapshots (``peak_mib`` key) are auto-detected and diffed with a
    peak-RSS table; timing snapshots go through pytest-benchmark. The two
    can't be mixed in one call.

    Implementation note: typer/click don't have a clean idiom for "list-typed
    positional + pass-through", so this command parses ``ctx.args`` by hand
    — anything before the first flag is a snapshot path, everything after
    is forwarded.
    """
    # Snapshots come first; once we see a flag (``-x`` / ``--foo``) every
    # subsequent token is forwarded to pytest-benchmark. That way the value
    # of a flag like ``-k "build and basic"`` doesn't get mistaken for a path.
    snapshots: list[Path] = []
    extra: list[str] = []
    seen_flag = False
    for arg in ctx.args:
        if arg.startswith("-"):
            seen_flag = True
        if seen_flag:
            extra.append(arg)
        else:
            snapshots.append(Path(arg))

    if len(snapshots) < 2:
        _suggest_snapshots(
            f"compare needs at least two snapshot paths (got {len(snapshots)})."
        )
        raise typer.Exit(code=2)

    missing = [p for p in snapshots if not p.exists()]
    if missing:
        _suggest_snapshots(f"missing snapshots: {[str(p) for p in missing]}")
        raise typer.Exit(code=2)

    # Auto-detect the metric from the snapshots (memory snapshots carry a
    # ``peak_mib`` key; timing ones don't) and route accordingly — no
    # ``memory compare`` needed.
    import json

    is_memory = ["peak_mib" in json.loads(p.read_text()) for p in snapshots]
    if any(is_memory):
        if not all(is_memory):
            typer.secho(
                "can't compare memory and timing snapshots together",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        if len(snapshots) != 2:
            typer.secho(
                "memory compare takes exactly 2 snapshots",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=2)
        from benchmarks.memory import compare_snapshots

        compare_snapshots(snapshots[0], snapshots[1])
        return

    # Override pytest-benchmark's wide default table: ``--group-by=fullname``
    # gives each test its own (baseline, candidate) mini-table and
    # ``--columns=min,iqr`` shows the noise-floor time plus spread. Applied
    # only when the user didn't pass their own.
    if not any(a.startswith("--columns") for a in extra):
        extra.insert(0, "--columns=min,iqr")
    if not any(a.startswith("--sort") for a in extra):
        extra.insert(0, "--sort=min")
    if not any(a.startswith("--group-by") for a in extra):
        extra.insert(0, "--group-by=fullname")

    cmd = [
        sys.executable,
        "-m",
        "pytest_benchmark",
        "compare",
        *[str(p) for p in snapshots],
        *extra,
    ]
    typer.secho(f"$ {' '.join(cmd)}", fg=typer.colors.BRIGHT_BLACK)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise typer.Exit(code=result.returncode)
