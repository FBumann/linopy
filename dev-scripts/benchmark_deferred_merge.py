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
    Build a capacity expansion + dispatch model with independent regional
    subsystems, each containing:

    - **Conventional generators**: dispatch + capacity variables, capacity
      limits, ramping constraints, marginal costs.
    - **Renewable generators**: dispatch limited by time-varying capacity
      factors, zero marginal cost, investment cost for capacity.
    - **Storage units**: charge / discharge / state-of-charge variables,
      energy balance (SOC tracking), charge/discharge capacity limits,
      cycling cost in objective.
    - **Demand**: must be met per region per timestep.

    All dimensions are COMPLETELY DISJOINT across regions (each region has
    its own time and asset dimensions). The objective sums costs from all
    regions and all component types.

    On master, chaining these additions creates dense cross-product arrays.
    With DeferredLinearExpression, parts are stored lazily.

    Parameters
    ----------
    n_groups : int
        Number of independent regional subsystems.
    n_plants : int
        Conventional plants per region.
    n_timesteps : int
        Time steps per region.
    """
    m = Model()

    n_renewables = max(1, n_plants // 3)
    n_storage = max(1, n_plants // 5)

    cost_parts = []

    for g in range(n_groups):
        time_idx = pd.RangeIndex(n_timesteps, name=f"time_{g}")
        plants = pd.Index([f"g{g}_p{p}" for p in range(n_plants)], name=f"plant_{g}")
        renew = pd.Index([f"g{g}_r{r}" for r in range(n_renewables)], name=f"renew_{g}")
        stor = pd.Index([f"g{g}_s{s}" for s in range(n_storage)], name=f"stor_{g}")

        # ── Conventional generators ──
        gen = m.add_variables(lower=0, coords=[time_idx, plants], name=f"gen_{g}")
        gen_cap = m.add_variables(lower=0, coords=[plants], name=f"gen_cap_{g}")

        # Capacity limit
        m.add_constraints(gen - gen_cap, "<=", 0, name=f"gen_cap_limit_{g}")

        # Ramping constraint: |gen(t) - gen(t-1)| <= 0.3 * cap
        # Approximate with two one-sided constraints
        gen_diff = gen.diff(f"time_{g}")
        ramp_limit = 0.3 * gen_cap
        m.add_constraints(gen_diff - ramp_limit, "<=", 0, name=f"ramp_up_{g}")
        m.add_constraints(-gen_diff - ramp_limit, "<=", 0, name=f"ramp_dn_{g}")

        # Marginal cost
        mc = pd.Series(10.0 + g * 5.0 + np.arange(n_plants, dtype=float), index=plants)
        cost_parts.append(gen * mc)

        # Investment cost for capacity
        inv_cost_gen = pd.Series(
            100.0 + np.arange(n_plants, dtype=float) * 10, index=plants
        )
        cost_parts.append(gen_cap * inv_cost_gen)

        # ── Renewables ──
        ren_gen = m.add_variables(lower=0, coords=[time_idx, renew], name=f"ren_{g}")
        ren_cap = m.add_variables(lower=0, coords=[renew], name=f"ren_cap_{g}")

        # Time-varying capacity factor (e.g. solar/wind profile)
        rng = np.random.default_rng(g * 100)
        cf = pd.DataFrame(
            rng.uniform(0.1, 0.9, (n_timesteps, n_renewables)),
            index=time_idx,
            columns=renew,
        )
        m.add_constraints(ren_gen - ren_cap * cf, "<=", 0, name=f"ren_cf_{g}")

        # Renewable investment cost (zero marginal cost)
        inv_cost_ren = pd.Series(
            200.0 + np.arange(n_renewables, dtype=float) * 20, index=renew
        )
        cost_parts.append(ren_cap * inv_cost_ren)

        # ── Storage ──
        charge = m.add_variables(lower=0, coords=[time_idx, stor], name=f"charge_{g}")
        discharge = m.add_variables(
            lower=0, coords=[time_idx, stor], name=f"discharge_{g}"
        )
        soc = m.add_variables(lower=0, coords=[time_idx, stor], name=f"soc_{g}")
        stor_cap = m.add_variables(lower=0, coords=[stor], name=f"stor_cap_{g}")

        # Charge / discharge capacity limits
        m.add_constraints(charge - 0.5 * stor_cap, "<=", 0, name=f"charge_lim_{g}")
        m.add_constraints(
            discharge - 0.5 * stor_cap, "<=", 0, name=f"discharge_lim_{g}"
        )

        # SOC capacity limit
        m.add_constraints(soc - stor_cap, "<=", 0, name=f"soc_lim_{g}")

        # Energy balance: soc(t) = soc(t-1) + 0.95*charge(t) - discharge(t)
        soc_prev = soc.shift({f"time_{g}": 1})
        m.add_constraints(
            soc - soc_prev - 0.95 * charge + discharge,
            "=",
            0,
            name=f"soc_balance_{g}",
        )

        # Storage cycling cost (small penalty for throughput)
        cost_parts.append(0.5 * charge + 0.5 * discharge)

        # Storage investment cost
        inv_cost_stor = pd.Series(
            150.0 + np.arange(n_storage, dtype=float) * 15, index=stor
        )
        cost_parts.append(stor_cap * inv_cost_stor)

        # ── Demand balance ──
        demand = pd.Series(np.full(n_timesteps, 100.0), index=time_idx)
        supply = (
            gen.sum(f"plant_{g}")
            + ren_gen.sum(f"renew_{g}")
            + discharge.sum(f"stor_{g}")
            - charge.sum(f"stor_{g}")
        )
        m.add_constraints(supply, ">=", demand, name=f"demand_{g}")

    # ── Objective: sum of all cost components across all regions ──
    # This is how a user would naturally write it: sum all parts.
    # Within a region, parts share time_g but have different asset dims.
    # Across regions, all dims are disjoint.
    # On master: merging creates dense cross-products.
    # On deferred: parts are stored lazily. The objective setter sums
    # to scalar, which works per-part when dims are disjoint.
    total_cost = cost_parts[0]
    for c in cost_parts[1:]:
        total_cost = total_cost + c
    m.add_objective(total_cost)

    return m


def build_shared_dims_model(
    n_components: int, n_assets: int, n_timesteps: int
) -> Model:
    """
    Build a model where ALL components share the SAME dimensions.

    Every variable has dims (time, asset) with identical coords.
    Additions are always same-shape → no DeferredLinearExpression created.
    This verifies there is no performance regression for the common case.

    Parameters
    ----------
    n_components : int
        Number of variable groups added to the objective.
    n_assets : int
        Assets (shared across all components).
    n_timesteps : int
        Time steps (shared across all components).
    """
    m = Model()

    time_idx = pd.RangeIndex(n_timesteps, name="time")
    assets = pd.Index([f"a{a}" for a in range(n_assets)], name="asset")

    cost_parts = []

    for c in range(n_components):
        gen = m.add_variables(lower=0, coords=[time_idx, assets], name=f"gen_{c}")
        cap = m.add_variables(lower=0, coords=[assets], name=f"cap_{c}")

        m.add_constraints(gen - cap, "<=", 0, name=f"cap_limit_{c}")

        mc = pd.Series(10.0 + c * 5.0 + np.arange(n_assets, dtype=float), index=assets)
        cost_parts.append(gen * mc)

    # All cost_parts have the same dims (time, asset) → same-shape fast path
    total_cost = cost_parts[0]
    for c in cost_parts[1:]:
        total_cost = total_cost + c

    m.add_objective(total_cost)

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".lp", delete=True) as f:
        m.to_file(f.name)

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


def run_disjoint_benchmarks() -> list[dict]:
    """Benchmark: disjoint regional subsystems (deferred helps)."""
    results = []

    scenarios = [
        # 2 groups, T=5, sweep P
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
        n_ren = max(1, n_plants // 3)
        n_stor = max(1, n_plants // 5)
        # Each group contributes dims: time, plant, renew, stor
        dims_per_group = n_ts * n_plants * n_ren * n_stor
        cross_size = dims_per_group**n_groups
        est_bytes = cross_size * 8 * 2
        if est_bytes > 500e6:
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


def run_shared_benchmarks() -> list[dict]:
    """Benchmark: shared dimensions (deferred NOT created, no benefit)."""
    results = []

    # (n_components, n_assets, n_timesteps)
    # All variables share (time, asset) dims → same-shape additions.
    # No DeferredLinearExpression is created. This verifies no regression.
    scenarios = [
        (2, 10, 10),
        (2, 30, 10),
        (2, 50, 10),
        (2, 100, 10),
        (5, 10, 10),
        (5, 30, 10),
        (5, 50, 10),
        (5, 100, 10),
        (10, 10, 10),
        (10, 30, 10),
        (10, 50, 10),
        (10, 100, 10),
    ]

    for n_comp, n_assets, n_ts in scenarios:
        stats = measure(
            lambda nc=n_comp, na=n_assets, nt=n_ts: build_shared_dims_model(nc, na, nt),
            warmup=1,
            repeats=5,
        )

        print(
            f"  C={n_comp:>2d} A={n_assets:>4d} T={n_ts:>3d}  "
            f"time={stats['time_median_s'] * 1000:>8.1f} ms  "
            f"peak={stats['peak_mb_median']:>8.1f} MB"
        )

        results.append(
            {
                "n_components": n_comp,
                "n_assets": n_assets,
                "n_timesteps": n_ts,
                "skipped": False,
                **stats,
            }
        )

    return results


def _plot_grouped_rows(
    axes,
    data_a,
    data_b,
    label_a,
    label_b,
    group_key,
    x_key,
    x_label,
    title_fn,
    start_row=0,
):
    """Plot rows of time/memory subplots grouped by `group_key`, swept by `x_key`."""
    c1, c2 = "#1f77b4", "#ff7f0e"

    def by_group(data, key):
        out: dict[int, list[dict]] = {}
        for r in data:
            if r.get("skipped"):
                continue
            out.setdefault(r[key], []).append(r)
        for g in out:
            out[g].sort(key=lambda x: x[x_key])
        return out

    groups_a = by_group(data_a, group_key)
    groups_b = by_group(data_b, group_key)
    all_g = sorted(set(groups_a) | set(groups_b))

    for i, g in enumerate(all_g):
        row = start_row + i
        ra = groups_a.get(g, [])
        rb = groups_b.get(g, [])

        xa = [r[x_key] for r in ra]
        xb = [r[x_key] for r in rb]

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
        ax.set_xlabel(x_label)
        ax.set_ylabel("Build time (ms)")
        ax.set_ylim(bottom=0)
        ax.set_title(title_fn(g, ra or rb))
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
        ax.set_xlabel(x_label)
        ax.set_ylabel("Peak memory (MB)")
        ax.set_ylim(bottom=0)
        ax.set_title(title_fn(g, ra or rb))
        ax.legend()
        ax.grid(True, alpha=0.3)

    return len(all_g)


def plot_comparison(file_a: str, file_b: str) -> None:
    import matplotlib.pyplot as plt

    with open(file_a) as f:
        data_a = json.load(f)
    with open(file_b) as f:
        data_b = json.load(f)

    label_a = data_a.get("label", Path(file_a).stem)
    label_b = data_b.get("label", Path(file_b).stem)

    disjoint_a = data_a.get("disjoint", [])
    disjoint_b = data_b.get("disjoint", [])
    shared_a = data_a.get("shared", [])
    shared_b = data_b.get("shared", [])

    # Count rows needed
    disjoint_groups = {
        r["n_groups"] for r in disjoint_a + disjoint_b if not r.get("skipped")
    }
    shared_groups = {
        r["n_components"] for r in shared_a + shared_b if not r.get("skipped")
    }
    n_rows = len(disjoint_groups) + len(shared_groups)

    if n_rows == 0:
        print("No data to plot.")
        return

    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), squeeze=False)
    fig.suptitle(
        f"Deferred Merge Benchmark: {label_a} vs {label_b}", fontsize=14, y=1.01
    )

    rows_used = _plot_grouped_rows(
        axes,
        disjoint_a,
        disjoint_b,
        label_a,
        label_b,
        group_key="n_groups",
        x_key="n_plants",
        x_label="Plants per region",
        title_fn=lambda g,
        rs: f"Disjoint regions: {g} groups, T={rs[0]['n_timesteps']}",
        start_row=0,
    )

    _plot_grouped_rows(
        axes,
        shared_a,
        shared_b,
        label_a,
        label_b,
        group_key="n_components",
        x_key="n_assets",
        x_label="Assets per component",
        title_fn=lambda g,
        rs: f"Shared time (no benefit): {g} components, T={rs[0]['n_timesteps']}",
        start_row=rows_used,
    )

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

    # ── Disjoint benchmarks ──
    print(f"Deferred merge benchmark (label={label!r})")
    print("=" * 80)
    print("\n[1/2] Disjoint regions (deferred helps)")
    print("  G=groups  P=plants/group  T=timesteps")
    print("-" * 80)
    disjoint = run_disjoint_benchmarks()

    # ── Shared benchmarks ──
    print("\n[2/2] Shared time dimension (no benefit expected)")
    print("  C=components  A=assets/component  T=timesteps")
    print("-" * 80)
    shared = run_shared_benchmarks()

    output = {"label": label, "disjoint": disjoint, "shared": shared}

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        print("\n(use -o FILE to save results for plotting)")


if __name__ == "__main__":
    main()
