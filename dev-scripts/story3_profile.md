# Story 3 Profile: Different Coordinate Subsets on Same Dimension

## Reproducer

`dev-scripts/story3.py` (22 effects, 40 of 500 contributors each, 500 timesteps):

```python
"""Story 3: Different coordinate subsets on same dimension."""
import tracemalloc
import numpy as np, linopy

tracemalloc.start()

def snap(label):
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  [{label}] current={cur / 1e6:.0f} MB  peak={peak / 1e6:.0f} MB")

m = linopy.Model()
rng = np.random.default_rng(42)
variables = []
for i in range(22):
    contribs = [f"c{c}" for c in sorted(rng.choice(500, 40, replace=False))]
    v = m.add_variables(coords=[contribs, range(500)],
                        dims=["contributor", "time"], name=f"share_e{i}")
    variables.append(v)
snap("after add_variables")

total = sum(variables)
snap("after sum(variables)")
print(f"  type: {type(total).__name__}")
if hasattr(total, "sizes"):
    print(f"  shape: {dict(total.sizes)}")

con = (total <= 1)
snap("after total <= 1")
tracemalloc.stop()
```

## How to run

```bash
python dev-scripts/story3.py
scalene run -o dev-scripts/story3_profile.json dev-scripts/story3.py
```

## Results (on master)

### tracemalloc snapshots (500 timesteps)

| Step                  | Current | Peak   |
|-----------------------|---------|--------|
| after `add_variables` | 17 MB   | 17 MB  |
| after `sum(variables)`| 24 MB   | 32 MB  |
| after `total <= 1`    | 24 MB   | 32 MB  |

Result shape: `{contributor: 40, time: 500, _term: 22}`

### tracemalloc snapshots (2000 timesteps)

| Step                  | Current | Peak    |
|-----------------------|---------|---------|
| after `add_variables` | 49 MB   | 49 MB   |
| after `sum(variables)`| 78 MB   | 109 MB  |
| after `total <= 1`    | 78 MB   | 109 MB  |

Result shape: `{contributor: 40, time: 2000, _term: 22}`

### Assessment

**Not a significant bottleneck.** The `contributor` dimension stays at 40 (per-expression
size), not the union of ~420 unique names. This is because `Variable.__add__` on master
already handles same-dim-name-different-coords by stacking into `_term` rather than
doing an outer join. Memory usage is modest and scales linearly with time.

This story may still become relevant if the result is later merged with another expression
that has a different `contributor` coordinate set — the outer join would then expand.
