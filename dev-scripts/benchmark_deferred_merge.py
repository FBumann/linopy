#!/usr/bin/env python3
"""
Benchmark: model building with disjoint-dimension expressions.

Measures time and peak memory for building a simplified energy dispatch model
where multiple technology groups have non-overlapping plant dimensions.
This is the core use case for DeferredLinearExpression.

Runs identically on master (eager merge) and feature/deferred-merge branches.

Usage:
    # Run benchmark, save results:
    python dev-scripts/benchmark_deferred_merge.py -o results.json --label "master"
    python dev-scripts/benchmark_deferred_merge.py -o results.json --label "deferred"

    # Plot comparison:
    python dev-scripts/benchmark_deferred_merge.py --plot master.json deferred.json
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from linopy import Model


def build_dispatch_model(n_groups: int, n_plants: int, n_timesteps: int) -> Model:
    """
    Build a dispatch model with independent regional subsystems.

    Each group represents an isolated region with its own buses/plants,
    modeled over a local time horizon. The dimensions are COMPLETELY DISJOINT
    across groups (each group has its own time and plant dimension).

    The objective sums the cost expressions from all groups. On master,
    chaining these additions creates a dense cross-product array of size
    n_plants^n_groups × n_timesteps^n_groups (exponential). With
    DeferredLinearExpression, parts are stored lazily.

    Parameters
    ----------
    n_groups : int
        Number of independent regional subsystems.
    n_plants : int
        Plants per region (each gets its own dimension).
    n_timesteps : int
        Time steps per region (each gets its own dimension).
    """
    m = Model()

    cost_parts = []

    for t in range(n_groups):
        time_idx = pd.RangeIndex(n_timesteps, name=f"time_{t}")
        plants = pd.Index([f"g{t}_p{p}" for p in range(n_plants)], name=f"plant_{t}")

        gen = m.add_variables(lower=0, coords=[time_idx, plants], name=f"gen_{t}")
        cap = m.add_variables(lower=0, coords=[plants], name=f"cap_{t}")

        # Capacity constraint (same-shape — no cross-product issue)
        m.add_constraints(gen - cap, "<=", 0, name=f"cap_limit_{t}")

        # Demand constraint per region
        demand = pd.Series(np.full(n_timesteps, 100.0), index=time_idx)
        m.add_constraints(gen.sum(f"plant_{t}"), ">=", demand, name=f"demand_{t}")

        # Marginal cost per plant
        mc = pd.Series(
            10.0 + t * 5.0 + np.arange(n_plants, dtype=float),
            index=plants,
        )
        cost_parts.append(gen * mc)

    # ── Objective: sum of cost across all regions ──
    # Chaining += on expressions with fully disjoint dims.
    # On master: each += calls merge() with join="outer" → exponential blowup.
    # On feature: DeferredLinearExpression stores parts lazily.
    total_cost = cost_parts[0]
    for c in cost_parts[1:]:
        total_cost = total_cost + c
    m.add_objective(total_cost)

    return m


def measure(func, warmup: int = 1, repeats: int = 5):
    """Run func, return timing and memory statistics."""
    # Warmup
    for _ in range(warmup):
        func()

    times = []
    peaks = []
    for _ in range(repeats):
        gc.collect()
        tracemalloc.start()
        t0 = time.perf_counter()
        func()
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed)
        peaks.append(peak / 1024 / 1024)

    return {
        "time_median_s": float(np.median(times)),
        "time_q25_s": float(np.percentile(times, 25)),
        "time_q75_s": float(np.percentile(times, 75)),
        "peak_mb_median": float(np.median(peaks)),
        "peak_mb_max": float(np.max(peaks)),
    }


def run_benchmarks() -> list[dict]:
    results = []

    # Scenarios grouped by n_groups, sweeping n_plants with fixed n_timesteps.
    # This gives clean scaling curves per group count.
    # Cross-product size = (n_timesteps * n_plants)^n_groups
    scenarios = [
        # 2 groups, T=10, sweep P
        (2, 5, 10),
        (2, 10, 10),
        (2, 20, 10),
        (2, 50, 10),
        (2, 100, 10),
        # 3 groups, T=10, sweep P
        (3, 5, 10),
        (3, 10, 10),
        (3, 15, 10),
        (3, 20, 10),
        # 4 groups, T=5, sweep P
        (4, 3, 5),
        (4, 5, 5),
        (4, 7, 5),
        (4, 10, 5),
        # 5 groups, T=3, sweep P
        (5, 2, 3),
        (5, 3, 3),
        (5, 4, 3),
        (5, 5, 3),
    ]

    for n_groups, n_plants, n_ts in scenarios:
        cross_size = (n_ts * n_plants) ** n_groups
        # Skip if cross-product would exceed ~8GB (to avoid OOM on master)
        est_bytes = cross_size * 8 * 2  # coeffs + vars arrays
        if est_bytes > 8e9:
            print(
                f"  SKIP  G={n_groups} P={n_plants:>3d} T={n_ts:>3d}  "
                f"(cross={cross_size:>14,} would need ~{est_bytes / 1e9:.0f} GB)"
            )
            results.append(
                {
                    "n_groups": n_groups,
                    "n_plants": n_plants,
                    "n_timesteps": n_ts,
                    "cross_product_size": cross_size,
                    "skipped": True,
                }
            )
            continue

        stats = measure(
            lambda ng=n_groups, np_=n_plants, nt=n_ts: build_dispatch_model(
                ng, np_, nt
            ),
            warmup=1,
            repeats=5,
        )

        print(
            f"  G={n_groups} P={n_plants:>3d} T={n_ts:>3d}  "
            f"cross={cross_size:>14,}  "
            f"time={stats['time_median_s'] * 1000:>8.1f} ms  "
            f"peak={stats['peak_mb_median']:>8.1f} MB"
        )

        results.append(
            {
                "n_groups": n_groups,
                "n_plants": n_plants,
                "n_timesteps": n_ts,
                "cross_product_size": cross_size,
                "skipped": False,
                **stats,
            }
        )

    return results


def plot_comparison(file_a: str, file_b: str) -> None:
    import matplotlib.pyplot as plt

    with open(file_a) as f:
        data_a = json.load(f)
    with open(file_b) as f:
        data_b = json.load(f)

    label_a = data_a.get("label", Path(file_a).stem)
    label_b = data_b.get("label", Path(file_b).stem)

    # Group results by n_groups
    def by_groups(data):
        out: dict[int, list[dict]] = {}
        for r in data["results"]:
            if r.get("skipped"):
                continue
            out.setdefault(r["n_groups"], []).append(r)
        # Sort each group by n_plants for a clean curve
        for g in out:
            out[g].sort(key=lambda x: x["n_plants"])
        return out

    groups_a = by_groups(data_a)
    groups_b = by_groups(data_b)
    all_g = sorted(set(groups_a) | set(groups_b))

    fig, axes = plt.subplots(len(all_g), 2, figsize=(12, 4 * len(all_g)), squeeze=False)
    fig.suptitle(
        f"Deferred Merge Benchmark: {label_a} vs {label_b}", fontsize=14, y=1.01
    )

    c1, c2 = "#1f77b4", "#ff7f0e"

    for row, g in enumerate(all_g):
        ra = groups_a.get(g, [])
        rb = groups_b.get(g, [])

        # x-axis: n_plants (the swept variable)
        xa = [r["n_plants"] for r in ra]
        xb = [r["n_plants"] for r in rb]

        # Time plot
        ax = axes[row][0]
        if ra:
            ta = [r["time_median_s"] * 1000 for r in ra]
            tq25 = [r["time_q25_s"] * 1000 for r in ra]
            tq75 = [r["time_q75_s"] * 1000 for r in ra]
            ax.fill_between(xa, tq25, tq75, color=c1, alpha=0.15)
            ax.plot(xa, ta, "o--", color=c1, label=label_a, alpha=0.8)
        if rb:
            tb = [r["time_median_s"] * 1000 for r in rb]
            tq25b = [r["time_q25_s"] * 1000 for r in rb]
            tq75b = [r["time_q75_s"] * 1000 for r in rb]
            ax.fill_between(xb, tq25b, tq75b, color=c2, alpha=0.15)
            ax.plot(xb, tb, "s-", color=c2, label=label_b, alpha=0.8)
        ax.set_xlabel("Plants per group")
        ax.set_ylabel("Build time (ms)")
        n_ts = ra[0]["n_timesteps"] if ra else (rb[0]["n_timesteps"] if rb else "?")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{g} groups, T={n_ts}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Memory plot
        ax = axes[row][1]
        if ra:
            ma = [r["peak_mb_median"] for r in ra]
            ax.plot(xa, ma, "o--", color=c1, label=label_a, alpha=0.8)
        if rb:
            mb = [r["peak_mb_median"] for r in rb]
            ax.plot(xb, mb, "s-", color=c2, label=label_b, alpha=0.8)
        ax.set_xlabel("Plants per group")
        ax.set_ylabel("Peak memory (MB)")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{g} groups, T={n_ts}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = "dev-scripts/benchmark_deferred_merge.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deferred merge benchmark")
    parser.add_argument("-o", "--output", help="Save results to JSON")
    parser.add_argument("--label", default=None, help="Label for this run")
    parser.add_argument(
        "--plot",
        nargs=2,
        metavar=("A", "B"),
        help="Plot comparison from two JSON files",
    )
    args = parser.parse_args()

    if args.plot:
        plot_comparison(args.plot[0], args.plot[1])
        return

    label = (
        args.label
        or subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
        ).strip()
    )

    print(f"Deferred merge benchmark (label={label!r})")
    print("  G=groups  P=plants/group  T=timesteps")
    print("  cross = (T × P)^G  (dense array size on master)")
    print("=" * 80)

    results = run_benchmarks()
    output = {"label": label, "results": results}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        print("\n(use -o FILE to save results for plotting)")


if __name__ == "__main__":
    main()
