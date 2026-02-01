"""Tests for default dtype settings (int32 labels, float32 floats)."""

import numpy as np
import pytest

from linopy import Model
from linopy.constants import DEFAULT_FLOAT_DTYPE, DEFAULT_LABEL_DTYPE

# --- int32 label tests ---


def test_default_label_dtype_is_int32():
    assert DEFAULT_LABEL_DTYPE == np.int32


def test_variable_labels_are_int32():
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    assert x.labels.dtype == np.int32


def test_constraint_labels_are_int32():
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    m.add_constraints(x >= 1, name="c")
    assert m.constraints["c"].labels.dtype == np.int32


def test_expression_vars_are_int32():
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    expr = 2 * x + 1
    assert expr.vars.dtype == np.int32


def test_solve_with_int32_labels():
    m = Model()
    x = m.add_variables(lower=0, upper=10, name="x")
    y = m.add_variables(lower=0, upper=10, name="y")
    m.add_constraints(x + y <= 15, name="c1")
    m.add_objective(x + 2 * y, sense="max")
    m.solve("highs")
    assert m.objective.value == pytest.approx(25.0)


def test_overflow_guard_variables():
    m = Model()
    m._xCounter = np.iinfo(np.int32).max - 1
    with pytest.raises(ValueError, match="exceeds the maximum"):
        m.add_variables(lower=0, upper=1, coords=[range(5)], name="x")


def test_overflow_guard_constraints():
    m = Model()
    x = m.add_variables(lower=0, upper=1, coords=[range(5)], name="x")
    m._cCounter = np.iinfo(np.int32).max - 1
    with pytest.raises(ValueError, match="exceeds the maximum"):
        m.add_constraints(x >= 0, name="c")


# --- float32 tests ---


def test_default_float_dtype_is_float32():
    assert DEFAULT_FLOAT_DTYPE == np.float32


def test_variable_bounds_are_float32():
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    assert x.lower.dtype == np.float32
    assert x.upper.dtype == np.float32


def test_expression_coeffs_are_float32():
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    expr = 2 * x + 1
    assert expr.coeffs.dtype == np.float32


def test_expression_const_is_float32():
    m = Model()
    x = m.add_variables(lower=0, upper=10, coords=[range(5)], name="x")
    expr = 2 * x + 1
    assert expr.const.dtype == np.float32


def test_solve_with_float32_model_data():
    m = Model()
    x = m.add_variables(lower=0, upper=10, name="x")
    y = m.add_variables(lower=0, upper=10, name="y")
    m.add_constraints(x + y <= 15, name="c1")
    m.add_objective(x + 2 * y, sense="max")
    m.solve("highs")
    assert m.objective.value == pytest.approx(25.0)


def test_solution_values_are_float64():
    m = Model()
    x = m.add_variables(lower=0, upper=10, name="x")
    m.add_objective(x, sense="max")
    m.solve("highs")
    assert m.solution["x"].dtype == np.float64
