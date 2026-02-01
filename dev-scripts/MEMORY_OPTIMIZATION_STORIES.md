# Linopy: Potential Memory / Performance Issues

Use cases that might cause problems due to how linopy uses xarray internally.
Each has a minimal reproducer. **Next step**: profile each one to verify.

---

## Story 1: `.sum()` Carries Dead Terms from Mask

A variable with a sparse `mask` has most labels set to `-1`. When `.sum(dim)`
stacks that dimension into `_term`, the dead entries come along.

```python
import numpy as np, xarray as xr, linopy

m = linopy.Model()

# Sparse mask: only 30 of 300 contributors active per effect
mask = xr.DataArray(np.zeros((300, 20), dtype=bool), dims=["contributor", "effect"])
rng = np.random.default_rng(42)
for e in range(20):
    mask.values[rng.choice(300, 30, replace=False), e] = True

var = m.add_variables(coords=[range(300), range(20), range(500)],
                      dims=["contributor", "effect", "time"],
                      name="share", mask=mask)

expr = var.sum("contributor")

print(expr.sizes["_term"])                        # 300 — not 30
print((expr.data.vars.values == -1).mean())       # ~90% dead
```

**Hypothesis**: `_term` should only contain the ~30 active contributors per
effect, not all 300. The 90% dead terms waste memory in every downstream op.

---

## Story 2: Cartesian Product from Disjoint Dimensions

Adding two expressions that share no dimensions (except e.g. `time`) causes
xarray to broadcast to the full Cartesian product.

```python
import linopy

m = linopy.Model()

x = m.add_variables(coords=[range(50), range(30), range(100)],
                    dims=["node", "line", "time"], name="x")
y = m.add_variables(coords=[range(40), range(20), range(100)],
                    dims=["vehicle", "route", "time"], name="y")

total = 2 * x + 3 * y  # disjoint dims: node/line vs vehicle/route

print(type(total).__name__)  # DeferredLinearExpression on feature branch

# But creating a constraint forces materialization:
# con = (total <= 1)  # → shape (50, 30, 40, 20, 100) = 120M elements
```

**Hypothesis**: The Cartesian product is unnecessary — both sides are
independent. The solver only needs the non-zero `(var_id, coeff)` pairs.

---

## Story 3: Different Coordinate Subsets on Same Dimension

Multiple expressions share a dimension name (`contributor`) but each has a
different subset of coordinates. Merging unions the coordinates and pads with
NaN.

```python
import numpy as np, linopy

m = linopy.Model()
rng = np.random.default_rng(42)

variables = []
for i in range(22):
    contribs = [f"c{c}" for c in sorted(rng.choice(500, 40, replace=False))]
    v = m.add_variables(coords=[contribs, range(100)],
                        dims=["contributor", "time"], name=f"share_e{i}")
    variables.append(v)

total = sum(variables)

print(type(total).__name__)
if hasattr(total, "sizes"):
    print(dict(total.sizes))  # contributor should be 40, not union of ~420
```

**Hypothesis**: After summing, the `contributor` coordinate should stay compact
(each expression's own subset), not expand to the union of all 420+ unique
contributor names.

---

## Story 4: `add_constraints` Broadcasts Before Masking

`model.add_constraints()` calls `xr.broadcast()` to align the constraint data
to a common shape, then applies the mask afterward. The full dense array is
allocated before masking.

```python
import numpy as np, xarray as xr, linopy

m = linopy.Model()
var = m.add_variables(coords=[range(200), range(20), range(100)],
                      dims=["contributor", "effect", "time"], name="x")

# Only 15% of (contributor, effect) pairs are active
mask = xr.DataArray(np.zeros((200, 20), dtype=bool), dims=["contributor", "effect"])
rng = np.random.default_rng(42)
for e in range(20):
    mask.values[rng.choice(200, 30, replace=False), e] = True

m.add_constraints(var <= 1, name="limit", mask=mask)

# xr.broadcast() creates full (200, 20, 100) array BEFORE mask is applied
```

**Hypothesis**: The full broadcast allocates memory for all 400K elements, then
mask sets 85% of labels to -1. Memory for the masked-out portion is wasted.

---

## How to Profile

Save any reproducer as e.g. `dev-scripts/story1.py`, then:

```bash
# Quick: just peak RSS and wall time
/usr/bin/time -l python dev-scripts/story1.py        # macOS
/usr/bin/time -v python dev-scripts/story1.py        # Linux

# Line-level memory + CPU attribution
scalene run -o story1_profile.json dev-scripts/story1.py
scalene view --cli story1_profile.json

# In-script memory snapshots (add to reproducer)
import tracemalloc
tracemalloc.start()
# ... code under test ...
current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1e6:.0f} MB, Peak: {peak / 1e6:.0f} MB")
tracemalloc.stop()
```

Tip: scale dimensions up/down to find the threshold where things break.

---

## Next Steps

1. **Profile each story** — pick realistic dimensions, measure actual numbers
2. **Identify which are real bottlenecks** vs acceptable overhead
3. **Brainstorm solutions** for the confirmed problems
