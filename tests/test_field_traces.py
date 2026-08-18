from __future__ import annotations

import numpy as np
import pytest

import hornlab_bempp_bem as bempp_bem
from hornlab_bempp_bem._constants import SPEED_OF_SOUND
from hornlab_bempp_bem.config import BIEFormulation, LinearSolver
from hornlab_bempp_bem.mesh import LoadedMesh
from hornlab_bempp_bem.result import MeshInfo


_FREQUENCIES = np.array([180.0, 320.0, 510.0], dtype=np.float64)
_POINTS = np.array(
    [
        [2.0, 1.7, 2.2],
        [-1.5, 2.1, 1.8],
        [1.6, -1.8, 2.4],
        [-1.7, -1.4, -2.2],
    ],
    dtype=np.float64,
)


def _tetrahedron_mesh(*, robin: bool = False) -> LoadedMesh:
    import bempp_cl.api as bempp_api

    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )
    tags = np.array([2, 8, 8, 8] if robin else [2, 1, 1, 1], dtype=np.int32)
    return LoadedMesh(
        grid=bempp_api.Grid(vertices.T, triangles.T),
        physical_tags=tags,
        info=MeshInfo(
            n_vertices=vertices.shape[0],
            n_triangles=triangles.shape[0],
            physical_groups=(
                {2: "source", 8: "robin"}
                if robin
                else {1: "rigid", 2: "source"}
            ),
            bounding_box_m=(vertices.min(axis=0), vertices.max(axis=0)),
        ),
    )


def _half_octahedron_mesh() -> LoadedMesh:
    import bempp_cl.api as bempp_api

    vertices = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1]],
        dtype=np.int32,
    )
    tags = np.array([2, 1, 1, 1], dtype=np.int32)
    return LoadedMesh(
        grid=bempp_api.Grid(vertices.T, triangles.T),
        physical_tags=tags,
        info=MeshInfo(
            n_vertices=vertices.shape[0],
            n_triangles=triangles.shape[0],
            physical_groups={1: "rigid", 2: "source"},
            bounding_box_m=(vertices.min(axis=0), vertices.max(axis=0)),
        ),
    )


def _observation() -> bempp_bem.ObservationConfig:
    return bempp_bem.ObservationConfig(
        planes=["probe"],
        angle_count=_POINTS.shape[0],
        custom_points={"probe": _POINTS},
    )


def _config(**kwargs) -> bempp_bem.SolveConfig:
    return bempp_bem.SolveConfig(
        observation=_observation(),
        velocity_sources={2: 1.0},
        solver=LinearSolver.LU,
        precision="double",
        assembly_backend="numba",
        return_surface_traces=True,
        **kwargs,
    )


def _evaluate_result(
    mesh: LoadedMesh,
    result: bempp_bem.SolveResult,
    *,
    symmetry_plane: str | None = None,
) -> np.ndarray:
    assert result.surface_pressure_complex is not None
    assert result.surface_neumann_complex is not None
    rows = []
    for index, frequency_hz in enumerate(result.frequencies_hz):
        k_real = 2.0 * np.pi * float(frequency_hz) / SPEED_OF_SOUND
        rows.append(
            bempp_bem.evaluate_exterior_from_traces(
                mesh,
                float(frequency_hz),
                k_real,
                result.surface_pressure_complex[index],
                result.surface_neumann_complex[index],
                result.observation_points.reshape(-1, 3),
                symmetry_plane=symmetry_plane,
            ).reshape(result.pressure_complex.shape[1:])
        )
    return np.stack(rows, axis=0)


def _assert_trace_parity(actual: np.ndarray, expected: np.ndarray) -> None:
    atol = 1.0e-10 * max(1.0, float(np.max(np.abs(expected))))
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=atol)


def test_return_surface_traces_defaults_off_and_requires_boolean():
    assert bempp_bem.SolveConfig().return_surface_traces is False
    with pytest.raises(ValueError, match="return_surface_traces"):
        bempp_bem.SolveConfig(return_surface_traces=1)  # type: ignore[arg-type]


@pytest.mark.slow
def test_rigid_multifrequency_traces_reproduce_observation_pressure():
    mesh = _tetrahedron_mesh()
    result = bempp_bem.solve_frequencies(mesh, _FREQUENCIES, _config())

    assert result.surface_pressure_complex is not None
    assert result.surface_neumann_complex is not None
    assert result.surface_pressure_complex.shape == (_FREQUENCIES.size, 4)
    assert result.surface_neumann_complex.shape == (_FREQUENCIES.size, 4)
    assert result.surface_pressure_complex.dtype == np.complex128
    assert result.surface_neumann_complex.dtype == np.complex128
    np.testing.assert_allclose(
        result.surface_neumann_complex[:, mesh.physical_tags == 2],
        -result.config.air_density,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(
        result.surface_neumann_complex[:, mesh.physical_tags == 1],
        0.0,
    )
    _assert_trace_parity(_evaluate_result(mesh, result), result.pressure_complex)


@pytest.mark.slow
def test_robin_total_neumann_passes_but_driver_only_fails_parity():
    mesh = _tetrahedron_mesh(robin=True)
    result = bempp_bem.solve_frequencies(
        mesh,
        _FREQUENCIES,
        _config(
            formulation=BIEFormulation.COMPLEX_K,
            complex_k_shift=0.05,
            impedance_sources={8: 0.45 + 0.12j},
        ),
    )

    total_field = _evaluate_result(mesh, result)
    _assert_trace_parity(total_field, result.pressure_complex)

    assert result.surface_pressure_complex is not None
    assert result.surface_neumann_complex is not None
    from hornlab_bempp_bem.bie import (
        _build_neumann_coefficients,
        _build_p1_to_dp0_projection,
        _setup_function_spaces,
    )

    p1_space, dp0_space = _setup_function_spaces(mesh.grid)
    projection = _build_p1_to_dp0_projection(p1_space, dp0_space)
    beta = np.zeros(dp0_space.global_dof_count, dtype=np.complex128)
    beta[mesh.physical_tags == 8] = result.config.impedance_sources[8]
    for index, frequency_hz in enumerate(result.frequencies_hz):
        omega = 2.0 * np.pi * float(frequency_hz)
        k = (
            omega
            / SPEED_OF_SOUND
            * (1.0 + 1j * result.config.complex_k_shift)
        )
        driver = _build_neumann_coefficients(
            dp0_space,
            mesh.physical_tags,
            omega,
            result.config,
            np.complex128,
            excluded_tags=result.config.impedance_sources,
            grid=mesh.grid,
        )
        expected_total = driver + (
            (1j * k)
            * beta
            * (projection @ result.surface_pressure_complex[index])
        )
        np.testing.assert_allclose(
            result.surface_neumann_complex[index],
            expected_total,
            rtol=1.0e-14,
            atol=1.0e-14,
        )

    driver_only = result.surface_neumann_complex.copy()
    driver_only[:, mesh.physical_tags == 8] = 0.0
    driver_rows = []
    for index, frequency_hz in enumerate(result.frequencies_hz):
        driver_rows.append(
            bempp_bem.evaluate_exterior_from_traces(
                mesh,
                float(frequency_hz),
                2.0 * np.pi * float(frequency_hz) / SPEED_OF_SOUND,
                result.surface_pressure_complex[index],
                driver_only[index],
                result.observation_points.reshape(-1, 3),
            ).reshape(result.pressure_complex.shape[1:])
        )
    driver_field = np.stack(driver_rows, axis=0)
    relative_error = np.linalg.norm(
        driver_field - result.pressure_complex
    ) / np.linalg.norm(result.pressure_complex)
    assert relative_error > 1.0e-2


@pytest.mark.slow
def test_native_symmetry_traces_use_reduced_dof_order_and_round_trip():
    mesh = _half_octahedron_mesh()
    frequency = _FREQUENCIES[:1]
    result = bempp_bem.solve_frequencies(
        mesh,
        frequency,
        _config(native_symmetry_plane="yz", require_closed_mesh=True),
    )

    assert result.surface_pressure_complex is not None
    assert result.surface_neumann_complex is not None
    assert result.surface_pressure_complex.shape == (1, 5)
    assert result.surface_neumann_complex.shape == (1, 4)
    _assert_trace_parity(
        _evaluate_result(mesh, result, symmetry_plane="yz"),
        result.pressure_complex,
    )
