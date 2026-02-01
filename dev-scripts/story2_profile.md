# Story 2 Profile: Cartesian Product from Disjoint Dimensions

## Reproducer

`dev-scripts/story2.py` (50 nodes x 30 lines x 100 time + 40 vehicles x 20 routes x 100 time):

```python
"""Story 2: Cartesian product from disjoint dimensions."""
import tracemalloc
import numpy as np, linopy

tracemalloc.start()

def snap(label):
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  [{label}] current={cur / 1e6:.0f} MB  peak={peak / 1e6:.0f} MB")

m = linopy.Model()
x = m.add_variables(coords=[range(50), range(30), range(100)],
                    dims=["node", "line", "time"], name="x")
y = m.add_variables(coords=[range(40), range(20), range(100)],
                    dims=["vehicle", "route", "time"], name="y")
snap("after add_variables")

total = 2 * x + 3 * y
snap("after 2*x + 3*y")
print(f"  type: {type(total).__name__}")

flat = total.flat
snap("after .flat")
print(f"  flat rows: {len(flat):,}")

con = (total <= 1)
snap("after total <= 1")
tracemalloc.stop()
```

## How to run

```bash
python dev-scripts/story2.py
scalene run -o dev-scripts/story2_profile.json dev-scripts/story2.py
```

## Results (on master — no DeferredLinearExpression)

### tracemalloc snapshots

| Step              | Current    | Peak       |
|-------------------|------------|------------|
| after `add_variables` | 13 MB  | 13 MB      |
| after `2*x + 3*y`    | 11,053 MB | 12,013 MB |
| after `.flat`         | 16,867 MB | 18,973 MB |
| after `total <= 1`   | 6,737 MB  | 18,973 MB |

### scalene top allocators (max footprint: 18,167 MB)

| File           | Line | Alloc     | What                                              |
|----------------|------|-----------|---------------------------------------------------|
| expressions.py | 2143 | 7,782 MB  | `xr.concat` const summation in `merge()` — Cartesian product |
| common.py      | 311  | 5,493 MB  | `v[mask]` in `to_dataframe` — filtering the 600M-element flat array |
| common.py      | 306  | 5,493 MB  | `v.values.reshape(-1)` in `to_dataframe` — flattening 600M elements |
| expressions.py | 1434 | 3,943 MB  | `df.groupby("vars").sum()` in `.flat` — groupby on huge DataFrame |
| expressions.py | 2141 | 3,666 MB  | `xr.concat` coeffs/vars in `merge()` — the join="outer" concat |
| expressions.py | 1309 | 918 MB    | `self.assign(const=...)` — scalar `+1` on Cartesian-product expr |
| expressions.py | 874  | 912 MB    | `assign_multiindex_safe` in `to_constraint` — allocating constraint data |

### Allocation chain

1. `2*x + 3*y`: `__add__` calls `merge()` with `join="outer"`. Since `x` has dims
   `(node, line, time)` and `y` has `(vehicle, route, time)`, xarray broadcasts to
   `(node, line, vehicle, route, time)` = 50×30×40×20×100 = **600M elements**.
   `xr.concat` allocates **7.8 GB + 3.7 GB** for this.

2. `.flat`: must flatten and filter the 600M-element arrays. `to_dataframe` allocates
   **5.5 GB** to reshape, then another **5.5 GB** to filter with mask. Groupby adds
   **3.9 GB**.

3. Only 230,000 of the 600M elements are non-zero (0.04%) — 99.96% waste.

### Root cause

`merge()` with `join="outer"` creates the full Cartesian product when expressions
have disjoint dimensions. The 600M-element dense array is created even though only
230K entries are meaningful. On the `feature/defered-merge` branch,
`DeferredLinearExpression` defers this, but materialization (for constraints) still
triggers the blowup.
