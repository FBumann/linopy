"""Story 1: .sum() carries dead terms from mask."""

import tracemalloc

import numpy as np
import xarray as xr

import linopy

tracemalloc.start()


def snap(label):
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  [{label}] current={cur / 1e6:.0f} MB  peak={peak / 1e6:.0f} MB")


m = linopy.Model()

# Sparse mask: only 30 of 300 contributors active per effect
mask = xr.DataArray(np.zeros((300, 20), dtype=bool), dims=["contributor", "effect"])
rng = np.random.default_rng(42)
for e in range(20):
    mask.values[rng.choice(300, 30, replace=False), e] = True

var = m.add_variables(
    coords=[range(300), range(20), range(2000)],
    dims=["contributor", "effect", "time"],
    name="share",
    mask=mask,
)
snap("after add_variables")

expr = var.sum("contributor")
snap("after .sum('contributor')")

print(f"  _term: {expr.sizes['_term']}")
print(f"  dead:  {(expr.data.vars.values == -1).mean():.0%}")

# Downstream: subtract from balance and add constraint
bal = m.add_variables(
    lower=0,
    upper=0,
    coords=[range(20), range(2000)],
    dims=["effect", "time"],
    name="bal",
)

lhs = bal - expr
snap("after bal - expr")

m.add_constraints(lhs == 0, name="balance")
snap("after add_constraints")

tracemalloc.stop()
print("done")
