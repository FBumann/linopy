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
    """Disjoint-dim Cartesian product + constraint."""
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
    m.add_constraints(2 * x + 3 * y <= 1, name="capacity")
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
        for n, t in [(5, 20), (10, 30), (10, 50), (20, 50)]
    ],
    "pypsa": [{"snapshots": s} for s in [10, 50, 100, 200]],
}


def benchmark_model(builder, kwargs):
    """Build a model and return peak memory (MB) and build time (s)."""
    gc.collect()
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
        # pypsa model
        nvars = getattr(model, "nvars", 0)
        ncons = getattr(model, "ncons", 0)

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


def plot_comparison(file_a, file_b):
    """4-panel comparison plot."""
    import matplotlib.pyplot as plt

    with open(file_a) as f:
        a = json.load(f)
    with open(file_b) as f:
        b = json.load(f)

    label_a = a.get("label", file_a)
    label_b = b.get("label", file_b)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Peak memory vs nvars (all models, log-log)
    ax = axes[0, 0]
    for data, label, marker in [(a, label_a, "o"), (b, label_b, "s")]:
        nvars, mem = [], []
        for mtype, runs in data["models"].items():
            for r in runs:
                nvars.append(r["nvars"])
                mem.append(r["peak_memory_mb"])
        ax.scatter(nvars, mem, label=label, marker=marker, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nvars")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Peak Memory vs Model Size")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Memory ratio (a/b) vs nvars
    ax = axes[0, 1]
    # Match runs by model type + params
    for mtype in set(a["models"]) & set(b["models"]):
        runs_a = {
            json.dumps(r["params"], sort_keys=True): r for r in a["models"][mtype]
        }
        runs_b = {
            json.dumps(r["params"], sort_keys=True): r for r in b["models"][mtype]
        }
        for key in set(runs_a) & set(runs_b):
            ra, rb = runs_a[key], runs_b[key]
            ratio = ra["peak_memory_mb"] / max(rb["peak_memory_mb"], 1e-6)
            ax.scatter(ra["nvars"], ratio, marker="o", alpha=0.7, label=mtype)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("nvars")
    ax.set_ylabel(f"Memory Ratio ({label_a} / {label_b})")
    ax.set_title("Memory Ratio")
    ax.grid(True, alpha=0.3)
    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    # Panel 3: Story 1 detail
    ax = axes[1, 0]
    for data, label, marker in [(a, label_a, "o"), (b, label_b, "s")]:
        if "story1" in data["models"]:
            runs = data["models"]["story1"]
            nvars = [r["nvars"] for r in runs]
            mem = [r["peak_memory_mb"] for r in runs]
            ax.plot(nvars, mem, marker=marker, label=label, alpha=0.7)
    ax.set_xlabel("nvars")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Story 1: Sparse Mask + .sum()")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 4: Story 2 detail
    ax = axes[1, 1]
    for data, label, marker in [(a, label_a, "o"), (b, label_b, "s")]:
        if "story2" in data["models"]:
            runs = data["models"]["story2"]
            nvars = [r["nvars"] for r in runs]
            mem = [r["peak_memory_mb"] for r in runs]
            ax.plot(nvars, mem, marker=marker, label=label, alpha=0.7)
    ax.set_xlabel("nvars")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Story 2: Disjoint-Dim Cartesian Product")
    ax.legend()
    ax.grid(True, alpha=0.3)

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
        "--plot", nargs=2, metavar="JSON", help="Compare two result JSON files"
    )
    args = parser.parse_args()

    if args.plot:
        plot_comparison(args.plot[0], args.plot[1])
        return

    print("Running model build benchmarks...")
    results = run_benchmarks(model_types=args.models, label=args.label)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
