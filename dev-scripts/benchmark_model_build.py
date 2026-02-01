"""
Benchmark peak memory during model building.

Usage:
    # Run and save results:
    python dev-scripts/benchmark_model_build.py -o results.json --label "my branch"

    # Compare two runs:
    python dev-scripts/benchmark_model_build.py --plot master.json pr.json
"""

import argparse
import gc
import json
import time
import tracemalloc

import numpy as np
import xarray as xr

import linopy

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def basic_model(n):
    """2×N² vars, 2×N² constraints."""
    m = linopy.Model()
    x = m.add_variables(coords=[range(n), range(n)], dims=["i", "j"], name="x")
    y = m.add_variables(coords=[range(n), range(n)], dims=["i", "j"], name="y")
    m.add_constraints(x + y <= 10, name="upper")
    m.add_constraints(x - y >= -5, name="lower")
    m.add_objective(x.sum() + 2 * y.sum())
    return m


def story1_model(n_contributors=300, n_effects=20, n_time=2000):
    """Sparse mask + .sum() dead terms + constraint."""
    m = linopy.Model()
    mask = xr.DataArray(
        np.zeros((n_contributors, n_effects), dtype=bool),
        dims=["contributor", "effect"],
    )
    rng = np.random.default_rng(42)
    active_per_effect = max(1, n_contributors // 10)
    for e in range(n_effects):
        mask.values[rng.choice(n_contributors, active_per_effect, replace=False), e] = (
            True
        )

    var = m.add_variables(
        coords=[range(n_contributors), range(n_effects), range(n_time)],
        dims=["contributor", "effect", "time"],
        name="share",
        mask=mask,
    )
    expr = var.sum("contributor")

    bal = m.add_variables(
        lower=0,
        upper=0,
        coords=[range(n_effects), range(n_time)],
        dims=["effect", "time"],
        name="bal",
    )
    m.add_constraints(bal - expr == 0, name="balance")
    m.add_objective(var.sum())
    return m


def story2_model(n_nodes=50, n_lines=30, n_vehicles=40, n_routes=20, n_time=100):
    """Disjoint-dim Cartesian product + constraint + objective (realistic user workflow)."""
    m = linopy.Model()
    x = m.add_variables(
        coords=[range(n_nodes), range(n_lines), range(n_time)],
        dims=["node", "line", "time"],
        name="x",
    )
    y = m.add_variables(
        coords=[range(n_vehicles), range(n_routes), range(n_time)],
        dims=["vehicle", "route", "time"],
        name="y",
    )
    total = 2 * x + 3 * y
    m.add_constraints(total <= 1, name="capacity")
    m.add_objective(x.sum() + y.sum())
    return m


def pypsa_model(snapshots=100):
    """PyPSA SciGrid-DE model. Returns None if pypsa not installed."""
    try:
        import pypsa
    except ImportError:
        return None

    n = pypsa.examples.scigrid_de()
    n.set_snapshots(n.snapshots[:snapshots])
    n.optimize.create_model()
    return n.model


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

MODEL_BUILDERS = {
    "basic": basic_model,
    "story1": story1_model,
    "story2": story2_model,
    "pypsa": pypsa_model,
}

SWEEP_SIZES = {
    "basic": [{"n": n} for n in [5, 10, 25, 50, 100, 200, 500]],
    "story1": [
        {"n_contributors": c, "n_effects": 20, "n_time": t}
        for c, t in [(30, 200), (100, 500), (300, 500), (300, 1000)]
    ],
    "story2": [
        {
            "n_nodes": n,
            "n_lines": int(n * 0.6),
            "n_vehicles": int(n * 0.8),
            "n_routes": int(n * 0.4),
            "n_time": t,
        }
        for n, t in [
            (5, 20),
            (10, 50),
            (20, 50),
            (20, 100),
            (30, 100),
            (40, 100),
            (50, 100),
        ]
    ],
    "pypsa": [{"snapshots": s} for s in [10, 50, 100, 200]],
}


_prev_model = [None]  # mutable container to hold ref for cleanup


def benchmark_model(builder, kwargs):
    """Build a model and return peak memory (MB) and build time (s)."""
    # Clean up previous model before measuring
    _prev_model[0] = None
    gc.collect()

    # Stop any lingering tracemalloc session, then start fresh
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    tracemalloc.start()
    tracemalloc.reset_peak()

    t0 = time.perf_counter()
    model = builder(**kwargs)
    elapsed = time.perf_counter() - t0

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if model is None:
        return None

    if isinstance(model, linopy.Model):
        nvars = model.nvars
        ncons = model.ncons
    else:
        nvars = getattr(model, "nvars", 0)
        ncons = getattr(model, "ncons", 0)

    # Keep model alive until next call so nvars/ncons are read before cleanup
    _prev_model[0] = model

    return {
        "peak_memory_mb": peak / 1e6,
        "build_time_s": elapsed,
        "nvars": int(nvars),
        "ncons": int(ncons),
        "params": kwargs,
    }


def run_benchmarks(model_types=None, label=""):
    """Run all benchmarks and return results dict."""
    if model_types is None:
        model_types = ["basic", "story1", "story2", "pypsa"]

    results = {"label": label, "models": {}}
    for mtype in model_types:
        builder = MODEL_BUILDERS[mtype]
        sizes = SWEEP_SIZES[mtype]
        runs = []
        for kwargs in sizes:
            print(f"  {mtype} {kwargs} ... ", end="", flush=True)
            res = benchmark_model(builder, kwargs)
            if res is None:
                print("skipped")
                continue
            runs.append(res)
            print(
                f"{res['peak_memory_mb']:.1f} MB, {res['build_time_s']:.2f}s, "
                f"{res['nvars']} vars, {res['ncons']} cons"
            )
        if runs:
            results["models"][mtype] = runs
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_comparison(files, focus="story2"):
    """3-panel comparison plot focused on a single model type."""
    import matplotlib.pyplot as plt

    datasets = []
    for fpath in files:
        with open(fpath) as f:
            d = json.load(f)
        d.setdefault("label", fpath)
        datasets.append(d)

    baseline = datasets[0]

    branch_colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]
    branch_linestyles = ["-", "--", ":"]
    ms = 10
    lw = 2.5

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Peak memory vs ncons
    ax = axes[0]
    for i, data in enumerate(datasets):
        if focus not in data["models"]:
            continue
        c = branch_colors[i % len(branch_colors)]
        ls = branch_linestyles[i % len(branch_linestyles)]
        runs = data["models"][focus]
        ncons = [r["ncons"] for r in runs]
        mem = [r["peak_memory_mb"] for r in runs]
        ax.plot(
            ncons,
            mem,
            marker="o",
            color=c,
            linestyle=ls,
            linewidth=lw,
            markersize=ms,
            alpha=0.85,
            label=data["label"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ncons")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title(f"{focus}: Peak Memory")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Memory ratio vs baseline
    ax = axes[1]
    if focus in baseline["models"]:
        runs_base = {
            json.dumps(r["params"], sort_keys=True): r
            for r in baseline["models"][focus]
        }
        for i, data in enumerate(datasets[1:], 1):
            if focus not in data["models"]:
                continue
            c = branch_colors[i % len(branch_colors)]
            ls = branch_linestyles[i % len(branch_linestyles)]
            runs_cur = {
                json.dumps(r["params"], sort_keys=True): r
                for r in data["models"][focus]
            }
            xs, ys, annots = [], [], []
            for key in sorted(set(runs_base) & set(runs_cur)):
                rb, rc = runs_base[key], runs_cur[key]
                ratio = rc["peak_memory_mb"] / max(rb["peak_memory_mb"], 1e-6)
                xs.append(rb["ncons"])
                ys.append(ratio)
                annots.append(f"{ratio:.2f}")
            ax.plot(
                xs,
                ys,
                marker="o",
                color=c,
                linestyle=ls,
                linewidth=lw,
                markersize=ms,
                alpha=0.85,
                label=data["label"],
            )
            for x, y, txt in zip(xs, ys, annots):
                ax.annotate(
                    txt,
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=8,
                    color=c,
                )
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1.5, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("ncons")
    ax.set_ylabel(f"Memory Ratio (vs {baseline['label']})")
    ax.set_title(f"{focus}: Memory Ratio")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: Build time vs ncons
    ax = axes[2]
    for i, data in enumerate(datasets):
        if focus not in data["models"]:
            continue
        c = branch_colors[i % len(branch_colors)]
        ls = branch_linestyles[i % len(branch_linestyles)]
        runs = data["models"][focus]
        ncons = [r["ncons"] for r in runs]
        times = [r["build_time_s"] for r in runs]
        ax.plot(
            ncons,
            times,
            marker="o",
            color=c,
            linestyle=ls,
            linewidth=lw,
            markersize=ms,
            alpha=0.85,
            label=data["label"],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ncons")
    ax.set_ylabel("Build Time (s)")
    ax.set_title(f"{focus}: Build Time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Model Build Benchmark — {focus}", fontsize=14, fontweight="bold")
    fig.tight_layout()
    plt.savefig("benchmark_model_build.png", dpi=150)
    print("Saved benchmark_model_build.png")
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark model build peak memory")
    parser.add_argument("-o", "--output", help="Save results to JSON file")
    parser.add_argument("--label", default="", help="Label for this run")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_BUILDERS.keys()),
        help="Which models to benchmark (default: all)",
    )
    parser.add_argument(
        "--plot",
        nargs="+",
        metavar="JSON",
        help="Compare result JSON files (first = baseline)",
    )
    args = parser.parse_args()

    if args.plot:
        plot_comparison(args.plot)
        return

    print("Running model build benchmarks...")
    results = run_benchmarks(model_types=args.models, label=args.label)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
