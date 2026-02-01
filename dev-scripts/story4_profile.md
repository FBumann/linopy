# Story 4 Profile: `add_constraints` Broadcasts Before Masking

## Reproducer

`dev-scripts/story4.py` (200 contributors, 20 effects, 500 timesteps, 15% mask density):

```python
"""Story 4: add_constraints broadcasts before masking."""
import tracemalloc
import numpy as np, xarray as xr, linopy

tracemalloc.start()

def snap(label):
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  [{label}] current={cur / 1e6:.0f} MB  peak={peak / 1e6:.0f} MB")

m = linopy.Model()
var = m.add_variables(coords=[range(200), range(20), range(500)],
                      dims=["contributor", "effect", "time"], name="x")
snap("after add_variables")

mask = xr.DataArray(np.zeros((200, 20), dtype=bool), dims=["contributor", "effect"])
rng = np.random.default_rng(42)
for e in range(20):
    mask.values[rng.choice(200, 30, replace=False), e] = True

m.add_constraints(var <= 1, name="limit", mask=mask)
snap("after add_constraints with mask")
tracemalloc.stop()
```

## How to run

```bash
python dev-scripts/story4.py
scalene run -o dev-scripts/story4_profile.json dev-scripts/story4.py
```

## Results (on master)

### tracemalloc snapshots (500 timesteps)

| Step                          | Current | Peak   |
|-------------------------------|---------|--------|
| after `add_variables`         | 54 MB   | 54 MB  |
| after `add_constraints`       | 102 MB  | 120 MB |

### tracemalloc snapshots (2000 timesteps)

| Step                          | Current | Peak   |
|-------------------------------|---------|--------|
| after `add_variables`         | 198 MB  | 198 MB |
| after `add_constraints`       | 390 MB  | 462 MB |

Mask density: 15%. Full shape: (200, 20, 2000) = 8M elements. Active: ~1.2M.

### Assessment

**Moderate overhead.** `add_constraints` allocates for the full 8M elements
(model.py:706 `xr.broadcast`), then masks 85% to label=-1. The ~2x memory
multiplier (198→462 MB) is from the constraint data being a copy of the
variable data plus labels/sign/rhs fields.

At this scale it's manageable. It would compound with Story 1 — if the expression
passed to `add_constraints` already carries dead terms from `.sum()`, the waste
multiplies.
