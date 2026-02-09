"""Tests for discontinuous piecewise linear constraints using binary formulation."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from linopy import Model, available_solvers
from linopy.constants import (
    DPWL_BINARY_SUFFIX,
    DPWL_DELTA_BOUND_SUFFIX,
    DPWL_DELTA_SUFFIX,
    DPWL_LINK_SUFFIX,
    DPWL_SELECT_SUFFIX,
)


class TestBasicSingleVariable:
    """Tests for single variable discontinuous piecewise constraints."""

    def test_basic_single_variable(self) -> None:
        """Test basic discontinuous PWL with a single variable."""
        m = Model()
        x = m.add_variables(name="x")

        # 3 pieces with gaps: [0,10], [15,25], [30,50]
        starts = xr.DataArray([0, 15, 30], dims=["piece"])
        ends = xr.DataArray([10, 25, 50], dims=["piece"])

        con = m.add_discontinuous_piecewise_constraints(x, starts, ends, dim="piece")

        # Check that variables and constraints were created
        assert f"dpwl0{DPWL_BINARY_SUFFIX}" in m.variables
        assert f"dpwl0{DPWL_DELTA_SUFFIX}" in m.variables
        assert f"dpwl0{DPWL_SELECT_SUFFIX}" in m.constraints
        assert f"dpwl0{DPWL_DELTA_BOUND_SUFFIX}" in m.constraints
        assert f"dpwl0{DPWL_LINK_SUFFIX}" in m.constraints

        # Check binary variable is indeed binary
        assert m.variables[f"dpwl0{DPWL_BINARY_SUFFIX}"].attrs.get("binary", False)

        # Check return value is the selection constraint
        assert con.name == f"dpwl0{DPWL_SELECT_SUFFIX}"

    def test_single_variable_with_coords(self) -> None:
        """Test single variable with explicit piece coordinates."""
        m = Model()
        x = m.add_variables(name="x")

        starts = xr.DataArray(
            [0, 15, 30], dims=["piece"], coords={"piece": ["low", "mid", "high"]}
        )
        ends = xr.DataArray(
            [10, 25, 50], dims=["piece"], coords={"piece": ["low", "mid", "high"]}
        )

        m.add_discontinuous_piecewise_constraints(x, starts, ends, dim="piece")

        # Check coordinates are preserved
        assert list(
            m.variables[f"dpwl0{DPWL_BINARY_SUFFIX}"].coords["piece"].values
        ) == [
            "low",
            "mid",
            "high",
        ]


class TestDictOfVariables:
    """Tests for dict of variables (linked quantities)."""

    def test_dict_of_variables(self) -> None:
        """Test discontinuous PWL with multiple linked variables."""
        m = Model()
        power = m.add_variables(name="power")
        cost = m.add_variables(name="cost")

        # Two operating regions with a gap
        starts = xr.DataArray(
            [[0, 60], [0, 150]],
            dims=["var", "piece"],
            coords={"var": ["power", "cost"]},
        )
        ends = xr.DataArray(
            [[50, 100], [100, 300]],
            dims=["var", "piece"],
            coords={"var": ["power", "cost"]},
        )

        m.add_discontinuous_piecewise_constraints(
            {"power": power, "cost": cost}, starts, ends, link_dim="var", dim="piece"
        )

        assert f"dpwl0{DPWL_BINARY_SUFFIX}" in m.variables
        assert f"dpwl0{DPWL_DELTA_SUFFIX}" in m.variables
        assert f"dpwl0{DPWL_LINK_SUFFIX}" in m.constraints

    def test_dict_with_additional_coords(self) -> None:
        """Test dict of variables with additional dimensions (e.g., generators)."""
        m = Model()
        generators = ["gen1", "gen2"]
        power = m.add_variables(coords=[generators], name="power")
        cost = m.add_variables(coords=[generators], name="cost")

        # Different pieces per generator
        starts = xr.DataArray(
            [
                [[0, 60], [0, 150]],  # gen1
                [[0, 40], [0, 80]],  # gen2
            ],
            dims=["gen", "var", "piece"],
            coords={"gen": generators, "var": ["power", "cost"]},
        )
        ends = xr.DataArray(
            [
                [[50, 100], [100, 300]],  # gen1
                [[30, 70], [60, 200]],  # gen2
            ],
            dims=["gen", "var", "piece"],
            coords={"gen": generators, "var": ["power", "cost"]},
        )

        m.add_discontinuous_piecewise_constraints(
            {"power": power, "cost": cost}, starts, ends, link_dim="var", dim="piece"
        )

        # Binary/delta variables should have gen and piece dimensions (not var)
        assert set(m.variables[f"dpwl0{DPWL_BINARY_SUFFIX}"].dims) == {"gen", "piece"}


class TestAutoDetectLinkDim:
    """Tests for auto-detection of link_dim."""

    def test_auto_detect_link_dim(self) -> None:
        """Test that link_dim is auto-detected from piece coordinates."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")

        starts = xr.DataArray(
            [[0, 15], [0, 20]],
            dims=["var", "piece"],
            coords={"var": ["x", "y"]},
        )
        ends = xr.DataArray(
            [[10, 25], [15, 30]],
            dims=["var", "piece"],
            coords={"var": ["x", "y"]},
        )

        # Should auto-detect link_dim="var" from coordinates matching dict keys
        m.add_discontinuous_piecewise_constraints(
            {"x": x, "y": y}, starts, ends, dim="piece"
        )

        assert f"dpwl0{DPWL_BINARY_SUFFIX}" in m.variables

    def test_auto_detect_fails_with_no_match(self) -> None:
        """Test that auto-detection fails with helpful error when no match."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")

        starts = xr.DataArray(
            [[0, 15], [0, 20]],
            dims=["other", "piece"],
            coords={"other": ["a", "b"]},  # doesn't match dict keys
        )
        ends = xr.DataArray(
            [[10, 25], [15, 30]],
            dims=["other", "piece"],
            coords={"other": ["a", "b"]},
        )

        with pytest.raises(ValueError, match="Could not auto-detect link_dim"):
            m.add_discontinuous_piecewise_constraints(
                {"x": x, "y": y}, starts, ends, dim="piece"
            )


class TestMasking:
    """Tests for masking functionality."""

    def test_nan_masking(self) -> None:
        """Test that NaN values in pieces create masked variables."""
        m = Model()
        x = m.add_variables(name="x")

        # Third piece is invalid (NaN)
        starts = xr.DataArray([0, 15, np.nan], dims=["piece"])
        ends = xr.DataArray([10, 25, np.nan], dims=["piece"])

        m.add_discontinuous_piecewise_constraints(x, starts, ends, dim="piece")

        # Check that binary variable has only 2 valid entries
        binary_labels = m.variables[f"dpwl0{DPWL_BINARY_SUFFIX}"].labels
        assert binary_labels.isel(piece=0).item() != -1
        assert binary_labels.isel(piece=1).item() != -1
        assert binary_labels.isel(piece=2).item() == -1

    def test_nan_masking_partial(self) -> None:
        """Test that piece is invalid if EITHER start OR end is NaN."""
        m = Model()
        x = m.add_variables(name="x")

        # Second piece has NaN end, third piece has NaN start
        starts = xr.DataArray([0, 15, np.nan], dims=["piece"])
        ends = xr.DataArray([10, np.nan, 50], dims=["piece"])

        m.add_discontinuous_piecewise_constraints(x, starts, ends, dim="piece")

        binary_labels = m.variables[f"dpwl0{DPWL_BINARY_SUFFIX}"].labels
        assert binary_labels.isel(piece=0).item() != -1  # valid
        assert binary_labels.isel(piece=1).item() == -1  # invalid (NaN end)
        assert binary_labels.isel(piece=2).item() == -1  # invalid (NaN start)

    def test_explicit_mask(self) -> None:
        """Test that explicit mask parameter works."""
        m = Model()
        x = m.add_variables(name="x")

        starts = xr.DataArray([0, 15, 30], dims=["piece"])
        ends = xr.DataArray([10, 25, 50], dims=["piece"])
        mask = xr.DataArray([True, False, True], dims=["piece"])

        m.add_discontinuous_piecewise_constraints(
            x, starts, ends, dim="piece", mask=mask
        )

        binary_labels = m.variables[f"dpwl0{DPWL_BINARY_SUFFIX}"].labels
        assert binary_labels.isel(piece=0).item() != -1
        assert binary_labels.isel(piece=1).item() == -1  # masked out
        assert binary_labels.isel(piece=2).item() != -1

    def test_skip_nan_check(self) -> None:
        """Test that skip_nan_check bypasses NaN detection."""
        m = Model()
        x = m.add_variables(name="x")

        # Has NaN but we skip the check
        starts = xr.DataArray([0, 15, np.nan], dims=["piece"])
        ends = xr.DataArray([10, 25, np.nan], dims=["piece"])

        m.add_discontinuous_piecewise_constraints(
            x, starts, ends, dim="piece", skip_nan_check=True
        )

        # All pieces should be created (no mask applied)
        binary_labels = m.variables[f"dpwl0{DPWL_BINARY_SUFFIX}"].labels
        assert binary_labels.isel(piece=0).item() != -1
        assert binary_labels.isel(piece=1).item() != -1
        assert binary_labels.isel(piece=2).item() != -1


class TestMultiDimensional:
    """Tests for multi-dimensional cases."""

    def test_multi_dimensional_pieces(self) -> None:
        """Test with pieces that vary across another dimension."""
        m = Model()
        generators = ["gen1", "gen2"]
        x = m.add_variables(coords=[generators], name="x")

        # Different pieces per generator
        starts = xr.DataArray(
            [[0, 60], [0, 40]],
            dims=["gen", "piece"],
            coords={"gen": generators},
        )
        ends = xr.DataArray(
            [[50, 100], [30, 70]],
            dims=["gen", "piece"],
            coords={"gen": generators},
        )

        m.add_discontinuous_piecewise_constraints(x, starts, ends, dim="piece")

        # Variables should have both gen and piece dimensions
        assert set(m.variables[f"dpwl0{DPWL_BINARY_SUFFIX}"].dims) == {"gen", "piece"}
        assert set(m.variables[f"dpwl0{DPWL_DELTA_SUFFIX}"].dims) == {"gen", "piece"}


class TestValidationErrors:
    """Tests for input validation error handling."""

    def test_invalid_expr_type(self) -> None:
        """Test that invalid expr type raises ValueError."""
        m = Model()

        starts = xr.DataArray([0, 15], dims=["piece"])
        ends = xr.DataArray([10, 25], dims=["piece"])

        with pytest.raises(ValueError, match="must be a Variable, LinearExpression"):
            m.add_discontinuous_piecewise_constraints(
                "invalid", starts, ends, dim="piece"
            )

    def test_missing_dim_in_starts(self) -> None:
        """Test error when dim is missing from piece_starts."""
        m = Model()
        x = m.add_variables(name="x")

        starts = xr.DataArray([0, 15], dims=["other"])
        ends = xr.DataArray([10, 25], dims=["piece"])

        with pytest.raises(ValueError, match="piece_starts must have dimension"):
            m.add_discontinuous_piecewise_constraints(x, starts, ends, dim="piece")

    def test_missing_dim_in_ends(self) -> None:
        """Test error when dim is missing from piece_ends."""
        m = Model()
        x = m.add_variables(name="x")

        starts = xr.DataArray([0, 15], dims=["piece"])
        ends = xr.DataArray([10, 25], dims=["other"])

        with pytest.raises(ValueError, match="piece_ends must have dimension"):
            m.add_discontinuous_piecewise_constraints(x, starts, ends, dim="piece")

    def test_mismatched_dimensions(self) -> None:
        """Test error when starts and ends have different dimensions."""
        m = Model()
        x = m.add_variables(name="x")

        starts = xr.DataArray([[0, 15]], dims=["extra", "piece"])
        ends = xr.DataArray([10, 25], dims=["piece"])

        with pytest.raises(ValueError, match="must have same dimensions"):
            m.add_discontinuous_piecewise_constraints(x, starts, ends, dim="piece")

    def test_expression_support(self) -> None:
        """Test that LinearExpression is supported as input."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")

        starts = xr.DataArray([0, 15], dims=["piece"])
        ends = xr.DataArray([10, 25], dims=["piece"])

        # Should not raise
        m.add_discontinuous_piecewise_constraints(x + y, starts, ends, dim="piece")

        assert f"dpwl0{DPWL_LINK_SUFFIX}" in m.constraints

    def test_link_dim_not_in_pieces(self) -> None:
        """Test error when link_dim is not in piece arrays."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")

        starts = xr.DataArray([0, 15], dims=["piece"])
        ends = xr.DataArray([10, 25], dims=["piece"])

        with pytest.raises(ValueError, match="not found in"):
            m.add_discontinuous_piecewise_constraints(
                {"x": x, "y": y}, starts, ends, link_dim="var", dim="piece"
            )

    def test_link_dim_coords_mismatch(self) -> None:
        """Test error when link_dim coords don't match dict keys."""
        m = Model()
        x = m.add_variables(name="x")
        y = m.add_variables(name="y")

        starts = xr.DataArray(
            [[0, 15], [0, 20]],
            dims=["var", "piece"],
            coords={"var": ["a", "b"]},  # doesn't match x, y
        )
        ends = xr.DataArray(
            [[10, 25], [15, 30]],
            dims=["var", "piece"],
            coords={"var": ["a", "b"]},
        )

        with pytest.raises(ValueError, match="don't match expression keys"):
            m.add_discontinuous_piecewise_constraints(
                {"x": x, "y": y}, starts, ends, link_dim="var", dim="piece"
            )


class TestNameGeneration:
    """Tests for automatic name generation."""

    def test_auto_name_generation(self) -> None:
        """Test that names are auto-generated with counter."""
        m = Model()
        x = m.add_variables(name="x")

        starts = xr.DataArray([0, 15], dims=["piece"])
        ends = xr.DataArray([10, 25], dims=["piece"])

        m.add_discontinuous_piecewise_constraints(x, starts, ends, dim="piece")
        m.add_discontinuous_piecewise_constraints(x, starts, ends, dim="piece")

        assert f"dpwl0{DPWL_BINARY_SUFFIX}" in m.variables
        assert f"dpwl1{DPWL_BINARY_SUFFIX}" in m.variables

    def test_custom_name(self) -> None:
        """Test that custom name parameter works."""
        m = Model()
        x = m.add_variables(name="x")

        starts = xr.DataArray([0, 15], dims=["piece"])
        ends = xr.DataArray([10, 25], dims=["piece"])

        m.add_discontinuous_piecewise_constraints(
            x, starts, ends, dim="piece", name="my_dpwl"
        )

        assert f"my_dpwl{DPWL_BINARY_SUFFIX}" in m.variables
        assert f"my_dpwl{DPWL_DELTA_SUFFIX}" in m.variables
        assert f"my_dpwl{DPWL_SELECT_SUFFIX}" in m.constraints


class TestSolverIntegration:
    """Integration tests with actual solver (requires Gurobi, CPLEX, or HiGHS)."""

    @pytest.mark.parametrize(
        ("solver", "module"),
        [("gurobi", "gurobipy"), ("cplex", "cplex"), ("highs", "highspy")],
    )
    def test_solve_single_variable_gap(self, solver: str, module: str) -> None:
        """Test solving with a gap - should find optimum in valid region."""
        if solver not in available_solvers:
            pytest.skip(f"{solver} not available")
        pytest.importorskip(module)

        m = Model()
        x = m.add_variables(name="x", lower=-100, upper=100)

        # Pieces: [0, 10] and [20, 30] with gap [10, 20]
        # If we maximize x, should get x=30
        starts = xr.DataArray([0.0, 20.0], dims=["piece"])
        ends = xr.DataArray([10.0, 30.0], dims=["piece"])

        m.add_discontinuous_piecewise_constraints(x, starts, ends, dim="piece")
        m.add_objective(x, sense="max")

        m.solve(solver_name=solver)

        assert m.status == "ok"
        assert np.isclose(m.solution["x"].item(), 30.0, atol=1e-5)

    @pytest.mark.parametrize(
        ("solver", "module"),
        [("gurobi", "gurobipy"), ("cplex", "cplex"), ("highs", "highspy")],
    )
    def test_solve_finds_correct_piece(self, solver: str, module: str) -> None:
        """Test that solver correctly selects the optimal piece."""
        if solver not in available_solvers:
            pytest.skip(f"{solver} not available")
        pytest.importorskip(module)

        m = Model()
        x = m.add_variables(name="x", lower=-100, upper=100)
        y = m.add_variables(name="y", lower=-100, upper=100)

        # x in [0, 10] -> y in [0, 5]
        # x in [20, 30] -> y in [10, 20]
        # Maximize y -> should pick second piece, y=20
        starts = xr.DataArray(
            [[0.0, 20.0], [0.0, 10.0]],
            dims=["var", "piece"],
            coords={"var": ["x", "y"]},
        )
        ends = xr.DataArray(
            [[10.0, 30.0], [5.0, 20.0]],
            dims=["var", "piece"],
            coords={"var": ["x", "y"]},
        )

        m.add_discontinuous_piecewise_constraints(
            {"x": x, "y": y}, starts, ends, link_dim="var", dim="piece"
        )
        m.add_objective(y, sense="max")

        m.solve(solver_name=solver)

        assert m.status == "ok"
        assert np.isclose(m.solution["y"].item(), 20.0, atol=1e-5)
        assert np.isclose(m.solution["x"].item(), 30.0, atol=1e-5)

    @pytest.mark.parametrize(
        ("solver", "module"),
        [("gurobi", "gurobipy"), ("cplex", "cplex"), ("highs", "highspy")],
    )
    def test_solve_step_function(self, solver: str, module: str) -> None:
        """Test solving with a step function (discontinuity in y)."""
        if solver not in available_solvers:
            pytest.skip(f"{solver} not available")
        pytest.importorskip(module)

        m = Model()
        x = m.add_variables(name="x", lower=0, upper=100)
        y = m.add_variables(name="y", lower=0, upper=100)

        # Step function: x in [0,50] -> y in [0,10], x in [50,100] -> y in [20,30]
        # Note: x=50 can belong to either piece
        starts = xr.DataArray(
            [[0.0, 50.0], [0.0, 20.0]],
            dims=["var", "piece"],
            coords={"var": ["x", "y"]},
        )
        ends = xr.DataArray(
            [[50.0, 100.0], [10.0, 30.0]],
            dims=["var", "piece"],
            coords={"var": ["x", "y"]},
        )

        m.add_discontinuous_piecewise_constraints(
            {"x": x, "y": y}, starts, ends, link_dim="var", dim="piece"
        )

        # Maximize y -> should pick second piece
        m.add_objective(y, sense="max")
        m.solve(solver_name=solver)

        assert m.status == "ok"
        assert np.isclose(m.solution["y"].item(), 30.0, atol=1e-5)
