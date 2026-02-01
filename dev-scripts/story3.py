"""Story 3: Different coordinate subsets on same dimension."""

import tracemalloc

import numpy as np

import linopy

tracemalloc.start()


def snap(label):
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  [{label}] current={cur / 1e6:.0f} MB  peak={peak / 1e6:.0f} MB")


m = linopy.Model()
rng = np.random.default_rng(42)

variables = []
for i in range(22):
    contribs = [f"c{c}" for c in sorted(rng.choice(500, 40, replace=False))]
    v = m.add_variables(
        coords=[contribs, range(500)],
        dims=["contributor", "time"],
        name=f"share_e{i}",
    )
    variables.append(v)
snap("after add_variables")

total = sum(variables)
snap("after sum(variables)")
print(f"  type: {type(total).__name__}")
if hasattr(total, "sizes"):
    print(f"  shape: {dict(total.sizes)}")

# Constraint
con = total <= 1
snap("after total <= 1")

tracemalloc.stop()
