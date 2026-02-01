"""Story 4: add_constraints broadcasts before masking."""

import tracemalloc

import numpy as np
import xarray as xr

import linopy

tracemalloc.start()


def snap(label):
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  [{label}] current={cur / 1e6:.0f} MB  peak={peak / 1e6:.0f} MB")


m = linopy.Model()

var = m.add_variables(
    coords=[range(200), range(20), range(500)],
    dims=["contributor", "effect", "time"],
    name="x",
)
snap("after add_variables")

# Only 15% of (contributor, effect) pairs are active
mask = xr.DataArray(np.zeros((200, 20), dtype=bool), dims=["contributor", "effect"])
rng = np.random.default_rng(42)
for e in range(20):
    mask.values[rng.choice(200, 30, replace=False), e] = True

m.add_constraints(var <= 1, name="limit", mask=mask)
snap("after add_constraints with mask")

tracemalloc.stop()
