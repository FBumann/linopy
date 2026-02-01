"""Story 2: Cartesian product from disjoint dimensions."""

import tracemalloc

import linopy

tracemalloc.start()


def snap(label):
    cur, peak = tracemalloc.get_traced_memory()
    print(f"  [{label}] current={cur / 1e6:.0f} MB  peak={peak / 1e6:.0f} MB")


m = linopy.Model()

x = m.add_variables(
    coords=[range(50), range(30), range(100)],
    dims=["node", "line", "time"],
    name="x",
)
y = m.add_variables(
    coords=[range(40), range(20), range(100)],
    dims=["vehicle", "route", "time"],
    name="y",
)
snap("after add_variables")

total = 2 * x + 3 * y
snap("after 2*x + 3*y")
print(f"  type: {type(total).__name__}")
if hasattr(total, "_parts"):
    print(f"  deferred: {len(total._parts)} parts")

# .flat — should work without materialization for disjoint dims
flat = total.flat
snap("after .flat")
print(f"  flat rows: {len(flat):,}")

# Constraint — forces materialization
con = total <= 1
snap("after total <= 1")

tracemalloc.stop()
