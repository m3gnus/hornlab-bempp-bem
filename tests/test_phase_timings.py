from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from hornlab_bempp_bem.config import SolveConfig


def test_solver_log_carries_phase_timings():
    from hornlab_bempp_bem.bie import FrequencyResult
    from hornlab_bempp_bem.sweep import _solver_log_entry

    result = FrequencyResult(
        frequency_hz=1000.0,
        pressure_on_surface=object(),
        neumann_data=object(),
        impedance=1.0 + 2.0j,
        iterations=None,
        converged=True,
        timing_s=0.75,
        phase_timings={"slp_assembly_s": 0.2, "linear_solve_s": 0.1},
        requested_precision="single",
        effective_precision="double",
    )

    entry = _solver_log_entry(result)

    assert entry["phase_timings"] == {
        "slp_assembly_s": 0.2,
        "linear_solve_s": 0.1,
    }
    assert entry["phase_timings"] is not result.phase_timings
    assert entry["requested_precision"] == "single"
    assert entry["effective_precision"] == "double"


def test_robin_solve_promotes_the_complete_precision_contract():
    from hornlab_bempp_bem.bie import solve_single_frequency

    grid = MagicMock()
    grid.number_of_elements = 2
    tags = np.array([1, 2], dtype=np.int32)
    p1_space = MagicMock()
    dp0_space = MagicMock()
    operator_precisions = []

    def capture_operator_kwargs(_backend, precision, *_args, **_kwargs):
        operator_precisions.append(precision)
        return {}

    with (
        patch(
            "hornlab_bempp_bem.bie.resolve_assembly_backend",
            return_value=SimpleNamespace(effective_backend="numba"),
        ),
        patch(
            "hornlab_bempp_bem.bie._operator_kwargs",
            side_effect=capture_operator_kwargs,
        ),
        patch(
            "hornlab_bempp_bem.bie._assemble_and_solve_impedance",
            return_value=(MagicMock(), MagicMock(), None, True, {}),
        ) as assemble,
        patch(
            "hornlab_bempp_bem.bie._compute_impedance",
            return_value=4.0 + 5.0j,
        ),
    ):
        result = solve_single_frequency(
            grid,
            tags,
            1000.0,
            SolveConfig(
                assembly_backend="numba",
                precision="single",
                impedance_sources={1: 0.05},
            ),
            p1_space=p1_space,
            dp0_space=dp0_space,
            closed_mesh_validated=True,
        )

    assert operator_precisions == ["double", "double"]
    impedance_config = assemble.call_args.args[6]
    assert impedance_config.precision == "double"
    assert result.requested_precision == "single"
    assert result.effective_precision == "double"


def test_single_frequency_reports_setup_core_and_impedance_phases():
    from hornlab_bempp_bem.bie import solve_single_frequency

    grid = MagicMock()
    grid.number_of_elements = 2
    tags = np.array([1, 2], dtype=np.int32)
    p1_space = MagicMock()
    dp0_space = MagicMock()
    core_timings = {
        "slp_assembly_s": 0.2,
        "dlp_assembly_s": 0.3,
        "linear_solve_s": 0.1,
    }

    with (
        patch(
            "hornlab_bempp_bem.bie.resolve_assembly_backend",
            return_value=SimpleNamespace(effective_backend="numba"),
        ),
        patch("hornlab_bempp_bem.bie._operator_kwargs", return_value={}),
        patch(
            "hornlab_bempp_bem.bie._build_neumann_data",
            return_value=MagicMock(),
        ),
        patch(
            "hornlab_bempp_bem.bie._assemble_and_solve",
            return_value=(MagicMock(), None, True, core_timings),
        ),
        patch(
            "hornlab_bempp_bem.bie._compute_impedance",
            return_value=4.0 + 5.0j,
        ),
    ):
        result = solve_single_frequency(
            grid,
            tags,
            1000.0,
            SolveConfig(
                assembly_backend="numba",
                restrict_neumann_space=False,
            ),
            p1_space=p1_space,
            dp0_space=dp0_space,
            closed_mesh_validated=True,
        )

    for name in (
        "function_spaces_s",
        "backend_setup_s",
        "neumann_data_s",
        "slp_assembly_s",
        "dlp_assembly_s",
        "linear_solve_s",
        "impedance_s",
        "total_s",
    ):
        assert name in result.phase_timings
        assert result.phase_timings[name] >= 0.0
    assert result.phase_timings["total_s"] == result.timing_s


def test_neumann_space_is_restricted_to_nonzero_elements():
    from hornlab_bempp_bem.bie import _restrict_neumann_to_nonzero_support

    full_space = MagicMock()
    neumann = SimpleNamespace(
        space=full_space,
        coefficients=np.array([0.0, 2.0 + 3.0j, 0.0, -1.0j]),
    )
    restricted_space = MagicMock()
    restricted_fun = MagicMock()

    with (
        patch(
            "bempp_cl.api.function_space",
            return_value=restricted_space,
        ) as function_space,
        patch(
            "bempp_cl.api.GridFunction",
            return_value=restricted_fun,
        ) as grid_function,
    ):
        actual_space, actual_fun = _restrict_neumann_to_nonzero_support(
            MagicMock(), neumann
        )

    assert actual_space is restricted_space
    assert actual_fun is restricted_fun
    np.testing.assert_array_equal(
        function_space.call_args.kwargs["support_elements"],
        [1, 3],
    )
    np.testing.assert_array_equal(
        grid_function.call_args.kwargs["coefficients"],
        [2.0 + 3.0j, -1.0j],
    )


def test_neumann_space_restriction_skips_empty_and_full_support():
    from hornlab_bempp_bem.bie import _restrict_neumann_to_nonzero_support

    for coefficients in (
        np.zeros(3, dtype=np.complex128),
        np.ones(3, dtype=np.complex128),
    ):
        full_space = MagicMock()
        neumann = SimpleNamespace(
            space=full_space,
            coefficients=coefficients,
        )
        with patch(
            "bempp_cl.api.function_space",
            side_effect=AssertionError("space should not be rebuilt"),
        ):
            actual_space, actual_fun = _restrict_neumann_to_nonzero_support(
                MagicMock(), neumann
            )
        assert actual_space is full_space
        assert actual_fun is neumann


def test_far_field_restricts_neumann_potential_to_nonzero_support():
    from hornlab_bempp_bem.bie import _evaluate_far_field

    p1_space = MagicMock()
    full_dp0 = MagicMock()
    full_dp0.grid = MagicMock()
    restricted_dp0 = MagicMock()
    pressure_fun = MagicMock()
    full_neumann = MagicMock()
    restricted_neumann = MagicMock()
    dlp_potential = MagicMock()
    slp_potential = MagicMock()
    dlp_potential.__mul__.return_value = np.array([2.0 + 0.0j])
    slp_potential.__mul__.return_value = np.array([0.5 + 0.0j])

    with (
        patch(
            "hornlab_bempp_bem.bie._restrict_neumann_to_nonzero_support",
            return_value=(restricted_dp0, restricted_neumann),
        ) as restrict,
        patch(
            "bempp_cl.api.operators.potential.helmholtz.double_layer",
            return_value=dlp_potential,
        ),
        patch(
            "bempp_cl.api.operators.potential.helmholtz.single_layer",
            return_value=slp_potential,
        ) as single_layer,
    ):
        pressure = _evaluate_far_field(
            p1_space,
            full_dp0,
            pressure_fun,
            full_neumann,
            2.0,
            np.array([[0.0, 0.0, 1.0]]),
            {},
        )

    restrict.assert_called_once_with(full_dp0.grid, full_neumann)
    assert single_layer.call_args.args[0] is restricted_dp0
    assert slp_potential.__mul__.call_args.args[0] is restricted_neumann
    np.testing.assert_array_equal(pressure, [1.5 + 0.0j])


def test_restrict_neumann_space_defaults_on_and_requires_boolean():
    assert SolveConfig().restrict_neumann_space is True

    try:
        SolveConfig(restrict_neumann_space=1)
    except ValueError as exc:
        assert "restrict_neumann_space must be a boolean" in str(exc)
    else:
        raise AssertionError("non-boolean restrict_neumann_space was accepted")
