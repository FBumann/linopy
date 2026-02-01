# Story 1 Profile: `.sum()` Carries Dead Terms from Mask

## Reproducer

`dev-scripts/story1.py` — scaled up from the snippet in the main doc
(300 contributors, 20 effects, **2000** timesteps):

```python
"""Story 1: .sum() carries dead terms from mask."""
import tracemalloc
import numpy as np, xarray as xr, linopy

tracemalloc.start()

def snap(label):
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  [{label}] current={cur / 1e6:.0f} MB  peak={peak / 1e6:.0f} MB")

m = linopy.Model()

mask = xr.DataArray(np.zeros((300, 20), dtype=bool), dims=["contributor", "effect"])
rng = np.random.default_rng(42)
for e in range(20):
    mask.values[rng.choice(300, 30, replace=False), e] = True

var = m.add_variables(coords=[range(300), range(20), range(2000)],
                      dims=["contributor", "effect", "time"],
                      name="share", mask=mask)
snap("after add_variables")

expr = var.sum("contributor")
snap("after .sum('contributor')")
print(f"  _term: {expr.sizes['_term']}")
print(f"  dead:  {(expr.data.vars.values == -1).mean():.0%}")

bal = m.add_variables(lower=0, upper=0, coords=[range(20), range(2000)],
                      dims=["effect", "time"], name="bal")
lhs = bal - expr
snap("after bal - expr")

m.add_constraints(lhs == 0, name="balance")
snap("after add_constraints")
tracemalloc.stop()
```

## How to run

```bash
# tracemalloc output (built into the script)
python dev-scripts/story1.py

# scalene line-level profiling
scalene run -o dev-scripts/story1_profile.json dev-scripts/story1.py
scalene view --cli dev-scripts/story1_profile.json

# parse scalene JSON for top allocators
python3 << 'EOF'
import json
with open("dev-scripts/story1_profile.json") as f:
    data = json.load(f)
print(f"Max footprint: {data['max_footprint_mb']:.0f} MB")
hotspots = []
for fpath, fdata in data["files"].items():
    short = fpath.split("/")[-1]
    for line in fdata["lines"]:
        alloc = line.get("n_malloc_mb", 0) or 0
        cpu = (line.get("n_cpu_percent_python", 0) or 0) + (line.get("n_cpu_percent_c", 0) or 0)
        if alloc > 2 or cpu > 3:
            hotspots.append({"file": short, "line": line["lineno"], "alloc_mb": alloc, "cpu": cpu})
hotspots.sort(key=lambda h: h["alloc_mb"], reverse=True)
for h in hotspots[:15]:
    print(f"  {h['file']:<22} line {h['line']:>5}  alloc={h['alloc_mb']:>7.1f} MB  cpu={h['cpu']:.1f}%")
EOF
```

## Results

### tracemalloc snapshots

| Step                        | Current | Peak   |
|-----------------------------|---------|--------|
| after `add_variables`       | 485 MB  | 486 MB |
| after `.sum("contributor")` | 594 MB  | 798 MB |
| after `bal - expr`          | 681 MB  | 897 MB |
| after `add_constraints`     | 682 MB  | 897 MB |

### scalene top allocators (max footprint: 926 MB)

| File           | Line | Alloc  | What                                                        |
|----------------|------|--------|-------------------------------------------------------------|
| common.py      | 266  | 280 MB | `DataArray(arr, ...)` — variable data creation              |
| common.py      | 517  | 275 MB | `ds.copy()` in `fill_missing_coords`                        |
| expressions.py | 1178 | 206 MB | `assign_multiindex_safe` after `.stack()` in `_sum()` — the bloated `_term=300` result |
| expressions.py | 2141 | 184 MB | `xr.concat` in `merge()` — the `bal - expr` subtraction     |
| expressions.py | 506  |  94 MB | `__neg__` — negating 300-term expression (90% dead)         |
| expressions.py | 365  |  92 MB | `.astype(float)` in `__init__` on bloated result            |
| model.py       | 537  |  92 MB | `add_variables` — label assignment                          |
| model.py       | 541  |  92 MB | `add_variables` — mask application                          |

### Allocation chain

1. `_sum()` (expressions.py:1172-1176): `.stack()` reshapes `_term=1, contributor=300`
   into `_term=300`. All 300 contributors become terms, including 270 dead ones (90%).
2. Line 1178: `assign_multiindex_safe` — **206 MB** to create the bloated Dataset.
3. `bal - expr` triggers `merge()` → `xr.concat` on the 300-term expression → **184 MB**.
4. `__neg__` inside subtraction → **94 MB** just to negate coefficients of 90% dead terms.

### Root cause

`_sum()` blindly stacks the full dimension into `_term` without filtering
`vars == -1`. Every downstream operation pays the cost of carrying 270 dead
terms per `(effect, time)` position.
