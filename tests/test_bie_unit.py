"""Unit tests for hornlab_bempp_bem.bie — air_density and surface_pressure_avg.

Tests that require bempp-cl are marked with pytest.mark.slow and skip if
bempp is unavailable. Pure-logic tests use mocks.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from types import SimpleNamespace

from hornlab_bempp_bem.bie import _choose_solver
from hornlab_bempp_bem.config import (
    LinearSolver,
    SolveConfig,
    SourceMotion,
    VelocityMode,
)

_FRAME_AXIS = np.array([0.0, 0.0, 1.0], dtype=np.float64)


def test_default_solver_remains_gmres_below_lu_threshold():
    config = SolveConfig(lu_threshold=100)

    assert config.solver is LinearSolver.GMRES
    assert _choose_solver(config, 1) is LinearSolver.GMRES


def test_auto_solver_opt_in_uses_lu_threshold():
    config = SolveConfig(solver=LinearSolver.AUTO, lu_threshold=100)

    assert _choose_solver(config, 100) is LinearSolver.LU
    assert _choose_solver(config, 101) is LinearSolver.GMRES


def test_explicit_solver_bypasses_lu_threshold():
    assert (
        _choose_solver(SolveConfig(solver=LinearSolver.GMRES), 1)
        is LinearSolver.GMRES
    )
    assert (
        _choose_solver(SolveConfig(solver=LinearSolver.LU), 100_000)
        is LinearSolver.LU
    )


def test_open_mesh_p1_space_excludes_free_boundary_dofs():
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
    closed_triangles = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )

    closed_grid = bempp_api.Grid(vertices.T, closed_triangles.T)
    closed_default = bempp_api.function_space(closed_grid, "P", 1)
    closed_all_vertices = bempp_api.function_space(
        closed_grid, "P", 1, include_boundary_dofs=True,
    )
    assert closed_default.global_dof_count == closed_all_vertices.global_dof_count
    np.testing.assert_array_equal(
        closed_default.local2global, closed_all_vertices.local2global,
    )
    np.testing.assert_array_equal(
        closed_default.local_multipliers,
        closed_all_vertices.local_multipliers,
    )

    open_grid = bempp_api.Grid(vertices.T, closed_triangles[:-1].T)
    open_default = bempp_api.function_space(open_grid, "P", 1)
    open_all_vertices = bempp_api.function_space(
        open_grid, "P", 1, include_boundary_dofs=True,
    )
    assert open_default.global_dof_count < open_all_vertices.global_dof_count
    assert np.any(np.asarray(open_default.local_multipliers) == 0.0)
    np.testing.assert_array_equal(
        open_all_vertices.local_multipliers,
        np.ones_like(open_all_vertices.local_multipliers),
    )


def _two_face_cap_grid(theta_rad: float) -> SimpleNamespace:
    """Two triangles whose outward unit normals sit at +/- theta from +z (equal
    area). The shared frame axis is +z, so each face projects to cos(theta);
    theta=0 is a flat disc (projection 1)."""
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    verts = np.array(
        [
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-c, 0.0, s],
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-c, 0.0, -s],
        ],
        dtype=np.float64,
    )
    elements = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return SimpleNamespace(vertices=verts.T, elements=elements.T)


def _two_tag_off_axis_grid(theta_rad: float) -> SimpleNamespace:
    """One +z source face and one equally sized face tilted from +z."""
    c = float(np.cos(theta_rad))
    s = float(np.sin(theta_rad))
    verts = np.array(
        [
            [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0 - c, 0.0, s],
        ],
        dtype=np.float64,
    )
    elements = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    return SimpleNamespace(vertices=verts.T, elements=elements.T)


# ---------------------------------------------------------------------------
# air_density passed into Neumann data
# ---------------------------------------------------------------------------

def _captured_coefficients():
    """Return a mock bempp_api whose GridFunction captures coefficients."""
    captured = {}

    def fake_grid_function(space, coefficients=None):
        gf = MagicMock()
        # Store a copy so we can inspect after the call
        captured["coefficients"] = coefficients.copy() if coefficients is not None else None
        gf.coefficients = captured["coefficients"]
        return gf

    return fake_grid_function, captured


class TestAirDensityInNeumann:

    def test_default_air_density_in_coefficients(self):
        fake_gf, captured = _captured_coefficients()
        with patch("bempp_cl.api.GridFunction", fake_gf):
            from hornlab_bempp_bem.bie import _build_neumann_data

            dp0_space = MagicMock()
            dp0_space.global_dof_count = 4

            tags = np.array([1, 2, 2, 1], dtype=np.int32)
            omega = 2 * np.pi * 1000.0
            config = SolveConfig(velocity_sources={2: 1.0})

            _build_neumann_data(dp0_space, tags, omega, config, "single")

            # Acceleration a*cos(omega t) under e^{-i omega t}: q = -rho*a
            # (momentum equation), frequency-independent.
            expected_coeff = -1.2041 * 1.0

            coeffs = captured["coefficients"]
            source_dofs = np.where(tags == 2)[0]
            for dof in source_dofs:
                np.testing.assert_allclose(coeffs[dof], expected_coeff, rtol=1e-6)

    def test_custom_air_density_propagates(self):
        fake_gf, captured = _captured_coefficients()
        with patch("bempp_cl.api.GridFunction", fake_gf):
            from hornlab_bempp_bem.bie import _build_neumann_data

            dp0_space = MagicMock()
            dp0_space.global_dof_count = 4

            tags = np.array([1, 2, 2, 1], dtype=np.int32)
            omega = 2 * np.pi * 500.0
            custom_rho = 1.18
            config = SolveConfig(
                velocity_sources={2: 1.0},
                air_density=custom_rho,
            )

            _build_neumann_data(dp0_space, tags, omega, config, "double")

            expected_coeff = -custom_rho * 1.0

            coeffs = captured["coefficients"]
            source_dofs = np.where(tags == 2)[0]
            for dof in source_dofs:
                np.testing.assert_allclose(coeffs[dof], expected_coeff, rtol=1e-6)

    def test_velocity_mode_velocity_uses_weight_directly(self):
        fake_gf, captured = _captured_coefficients()
        with patch("bempp_cl.api.GridFunction", fake_gf):
            from hornlab_bempp_bem.bie import _build_neumann_data

            dp0_space = MagicMock()
            dp0_space.global_dof_count = 3

            tags = np.array([2, 2, 1], dtype=np.int32)
            omega = 2 * np.pi * 2000.0
            config = SolveConfig(
                velocity_sources={2: 0.5},
                velocity_mode=VelocityMode.VELOCITY,
                air_density=1.2041,
            )

            _build_neumann_data(dp0_space, tags, omega, config, "single")

            expected_coeff = 1j * 1.2041 * omega * 0.5
            coeffs = captured["coefficients"]

            np.testing.assert_allclose(coeffs[0], expected_coeff, rtol=1e-6)
            np.testing.assert_allclose(coeffs[1], expected_coeff, rtol=1e-6)
            assert coeffs[2] == 0.0

    def test_zero_omega_acceleration_gives_zero(self):
        fake_gf, captured = _captured_coefficients()
        with patch("bempp_cl.api.GridFunction", fake_gf):
            from hornlab_bempp_bem.bie import _build_neumann_data

            dp0_space = MagicMock()
            dp0_space.global_dof_count = 2

            tags = np.array([2, 2], dtype=np.int32)
            config = SolveConfig(velocity_sources={2: 1.0})

            _build_neumann_data(dp0_space, tags, 0.0, config, "single")

            coeffs = captured["coefficients"]
            assert coeffs[0] == 0.0
            assert coeffs[1] == 0.0


# ---------------------------------------------------------------------------
# compute_surface_pressure_avg
# ---------------------------------------------------------------------------

class TestComputeSurfacePressureAvg:

    def test_impedance_uses_same_area_weighted_surface_average(self):
        from hornlab_bempp_bem.bie import _compute_impedance

        grid = MagicMock()
        grid.volumes = np.array([0.03, 0.01])
        p_surface = MagicMock()
        p_surface.coefficients = np.array(
            [100, 100, 100, 200, 200, 200], dtype=np.complex128,
        )
        p1_space = MagicMock()
        p1_space.local2global = np.array([[0, 1, 2], [3, 4, 5]])
        p1_space.local_multipliers = np.ones((2, 3))
        tags = np.array([2, 2], dtype=np.int32)

        expected = (100.0 * 0.03 + 200.0 * 0.01) / 0.04
        np.testing.assert_allclose(
            _compute_impedance(grid, p_surface, tags, p1_space, source_tag=2),
            expected,
        )
        assert (
            _compute_impedance(grid, p_surface, tags, p1_space, source_tag=3)
            == 0.0 + 0.0j
        )

    def test_single_tag_uniform_pressure(self):
        from hornlab_bempp_bem.bie import compute_surface_pressure_avg

        n_verts = 6

        grid = MagicMock()
        grid.elements = MagicMock()
        grid.elements.T = np.array([
            [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],
        ], dtype=np.int32)
        grid.volumes = np.array([0.01, 0.01, 0.01, 0.01])

        pressure_val = 100.0 + 20.0j
        coeffs = np.full(n_verts, pressure_val, dtype=np.complex128)
        p_surface = MagicMock()
        p_surface.coefficients = coeffs

        p1_space = MagicMock()
        p1_space.local2global = np.array([
            [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5],
        ])
        p1_space.local_multipliers = np.ones((4, 3))

        tags = np.array([2, 2, 2, 2], dtype=np.int32)

        result = compute_surface_pressure_avg(
            grid, p_surface, tags, p1_space, [2],
        )

        np.testing.assert_allclose(result[2], pressure_val, rtol=1e-10)

    def test_area_weighting_matters(self):
        from hornlab_bempp_bem.bie import compute_surface_pressure_avg

        grid = MagicMock()
        grid.elements = MagicMock()
        grid.elements.T = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        grid.volumes = np.array([0.03, 0.01])

        coeffs = np.array([100, 100, 100, 200, 200, 200], dtype=np.complex128)
        p_surface = MagicMock()
        p_surface.coefficients = coeffs

        p1_space = MagicMock()
        p1_space.local2global = np.array([[0, 1, 2], [3, 4, 5]])
        p1_space.local_multipliers = np.ones((2, 3))

        tags = np.array([2, 2], dtype=np.int32)

        result = compute_surface_pressure_avg(
            grid, p_surface, tags, p1_space, [2],
        )

        expected = (100.0 * 0.03 + 200.0 * 0.01) / 0.04
        np.testing.assert_allclose(result[2], expected, rtol=1e-10)

    def test_missing_tag_returns_zero(self):
        from hornlab_bempp_bem.bie import compute_surface_pressure_avg

        grid = MagicMock()
        grid.elements = MagicMock()
        grid.elements.T = np.array([[0, 1, 2]], dtype=np.int32)
        grid.volumes = np.array([0.01])

        p_surface = MagicMock()
        p_surface.coefficients = np.array([1.0, 1.0, 1.0], dtype=np.complex128)

        p1_space = MagicMock()
        p1_space.local2global = np.array([[0, 1, 2]])
        p1_space.local_multipliers = np.ones((1, 3))

        tags = np.array([1], dtype=np.int32)

        result = compute_surface_pressure_avg(
            grid, p_surface, tags, p1_space, [2, 3],
        )

        assert result[2] == 0.0 + 0.0j
        assert result[3] == 0.0 + 0.0j

    def test_multiple_tags_independent(self):
        from hornlab_bempp_bem.bie import compute_surface_pressure_avg

        grid = MagicMock()
        grid.elements = MagicMock()
        grid.elements.T = np.array([
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
        ], dtype=np.int32)
        grid.volumes = np.array([0.01, 0.01, 0.01])

        coeffs = np.array(
            [50, 50, 50, 150, 150, 150, 50, 50, 50], dtype=np.complex128,
        )
        p_surface = MagicMock()
        p_surface.coefficients = coeffs

        p1_space = MagicMock()
        p1_space.local2global = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        p1_space.local_multipliers = np.ones((3, 3))

        tags = np.array([2, 3, 2], dtype=np.int32)

        result = compute_surface_pressure_avg(
            grid, p_surface, tags, p1_space, [2, 3],
        )

        np.testing.assert_allclose(result[2], 50.0, rtol=1e-10)
        np.testing.assert_allclose(result[3], 150.0, rtol=1e-10)

    def test_excluded_boundary_dofs_contribute_zero(self):
        from hornlab_bempp_bem.bie import compute_surface_pressure_avg

        grid = MagicMock()
        grid.volumes = np.array([1.0, 1.0])
        p_surface = MagicMock()
        p_surface.coefficients = np.array([30.0, 60.0], dtype=np.complex128)
        p1_space = MagicMock()
        p1_space.local2global = np.array([[0, 1, 0], [1, 0, 0]])
        p1_space.local_multipliers = np.array([
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

        result = compute_surface_pressure_avg(
            grid,
            p_surface,
            np.array([2, 2], dtype=np.int32),
            p1_space,
            [2],
        )

        np.testing.assert_allclose(result[2], 25.0)


# ---------------------------------------------------------------------------
# Robin contribution assembly
# ---------------------------------------------------------------------------

class TestRobinContribution:

    def test_projection_respects_excluded_boundary_dofs(self):
        from hornlab_bempp_bem.bie import _build_p1_to_dp0_projection

        p1_space = SimpleNamespace(
            global_dof_count=2,
            local2global=np.array([[0, 1, 0], [1, 0, 0]]),
            local_multipliers=np.array([
                [1.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]),
        )
        dp0_space = SimpleNamespace(global_dof_count=2)

        projection = _build_p1_to_dp0_projection(p1_space, dp0_space)

        np.testing.assert_allclose(
            projection.toarray(),
            np.array([
                [1.0 / 3.0, 1.0 / 3.0],
                [0.0, 1.0 / 3.0],
            ]),
        )

    def test_sparse_projection_is_not_densified(self):
        import scipy.sparse as sp

        from hornlab_bempp_bem.bie import _build_robin_contribution

        single_layer = np.array(
            [
                [1.0 + 2.0j, 3.0 - 1.0j, -2.0 + 0.5j],
                [0.5 - 0.25j, -1.0 + 4.0j, 2.0 + 3.0j],
            ],
            dtype=np.complex128,
        )
        beta = np.array([0.05, 0.0, 0.2j], dtype=np.complex128)
        projection_dense = np.array(
            [
                [0.5, 0.5],
                [1.0, 0.0],
                [0.25, 0.75],
            ],
            dtype=np.float64,
        )
        projection = sp.csr_matrix(projection_dense)
        k = 2.5 + 0.1j
        expected = (
            (1j * k)
            * single_layer
            @ np.diag(beta)
            @ projection_dense
        )

        with patch.object(
            sp.csr_matrix,
            "toarray",
            side_effect=AssertionError("sparse projection was densified"),
        ):
            actual = _build_robin_contribution(
                single_layer, beta, projection, k,
            )

        np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)


# ---------------------------------------------------------------------------
# Axial (rigid-piston) source motion (parity with hornlab-metal-bem)
# ---------------------------------------------------------------------------


class TestAxialElementScale:

    def test_flat_disc_projects_to_unity(self):
        from hornlab_bempp_bem.bie import _build_axial_element_scale

        grid = _two_face_cap_grid(0.0)
        tags = np.array([2, 2], dtype=np.int32)
        scale = _build_axial_element_scale(grid, tags, [2], _FRAME_AXIS)
        np.testing.assert_allclose(scale, [1.0, 1.0], atol=1e-12)

    def test_tilted_cap_projects_to_cos_theta(self):
        from hornlab_bempp_bem.bie import _build_axial_element_scale

        theta = np.deg2rad(35.0)
        grid = _two_face_cap_grid(theta)
        tags = np.array([2, 2], dtype=np.int32)
        scale = _build_axial_element_scale(grid, tags, [2], _FRAME_AXIS)
        np.testing.assert_allclose(scale, [np.cos(theta)] * 2, rtol=1e-9)

    def test_non_source_and_absent_tag(self):
        from hornlab_bempp_bem.bie import _build_axial_element_scale

        grid = _two_face_cap_grid(np.deg2rad(30.0))
        tags = np.array([2, 7], dtype=np.int32)
        scale = _build_axial_element_scale(grid, tags, [2], _FRAME_AXIS)
        # A lone tilted face still projects onto the shared frame axis.
        np.testing.assert_allclose(scale[0], np.cos(np.deg2rad(30.0)), rtol=1e-9)
        assert scale[1] == 0.0
        # No source faces -> None (caller keeps the normal BC).
        assert _build_axial_element_scale(grid, tags, [99], _FRAME_AXIS) is None

    def test_secondary_off_axis_tag_uses_shared_frame_axis(self):
        """Match metal semantics: one frame axis, with a sign vote per tag."""
        from hornlab_bempp_bem.bie import _build_axial_element_scale

        theta = np.deg2rad(45.0)
        grid = _two_tag_off_axis_grid(theta)
        tags = np.array([2, 3], dtype=np.int32)

        forward = _build_axial_element_scale(grid, tags, [2, 3], _FRAME_AXIS)
        reversed_axis = _build_axial_element_scale(
            grid, tags, [2, 3], -_FRAME_AXIS
        )

        expected = np.array([1.0, np.cos(theta)])
        np.testing.assert_allclose(forward, expected, rtol=1e-12)
        np.testing.assert_allclose(reversed_axis, expected, rtol=1e-12)


class TestAxialNeumannData:

    def test_precomputed_element_scale_skips_geometry_rebuild(self):
        from hornlab_bempp_bem.bie import _build_neumann_coefficients

        tags = np.array([2, 2], dtype=np.int32)
        omega = 2 * np.pi * 1000.0
        config = SolveConfig(
            velocity_sources={2: 1.0},
            velocity_mode=VelocityMode.VELOCITY,
            source_motion=SourceMotion.AXIAL,
        )
        axial_element_scale = np.array([0.25, 1.0])

        with patch(
            "hornlab_bempp_bem.bie._build_axial_element_scale",
            side_effect=AssertionError("geometry projection was rebuilt"),
        ):
            coeffs = _build_neumann_coefficients(
                SimpleNamespace(global_dof_count=2),
                tags,
                omega,
                config,
                np.complex128,
                axial_element_scale=axial_element_scale,
            )

        expected = 1j * config.air_density * omega * axial_element_scale
        np.testing.assert_allclose(coeffs, expected, rtol=1e-12)

    def test_robin_driver_coefficients_use_shared_frame_axis(self):
        from hornlab_bempp_bem.bie import _build_neumann_coefficients

        theta = np.deg2rad(45.0)
        grid = _two_tag_off_axis_grid(theta)
        tags = np.array([2, 3], dtype=np.int32)
        omega = 2 * np.pi * 1000.0
        config = SolveConfig(
            velocity_sources={2: 1.0, 3: 1.0},
            velocity_mode=VelocityMode.VELOCITY,
            source_motion=SourceMotion.AXIAL,
            impedance_sources={1: 0.05},
        )

        coeffs = _build_neumann_coefficients(
            SimpleNamespace(global_dof_count=2),
            tags,
            omega,
            config,
            np.complex128,
            excluded_tags=config.impedance_sources,
            grid=grid,
            source_axis=_FRAME_AXIS,
        )

        expected = 1j * config.air_density * omega * np.array(
            [1.0, np.cos(theta)]
        )
        np.testing.assert_allclose(coeffs, expected, rtol=1e-12)

    def test_robin_driver_coefficients_exclude_impedance_tags(self):
        from hornlab_bempp_bem.bie import _build_neumann_coefficients

        tags = np.array([2, 3], dtype=np.int32)
        omega = 2 * np.pi * 1000.0
        config = SolveConfig(
            velocity_sources={2: 1.0, 3: 1.0},
            velocity_mode=VelocityMode.VELOCITY,
            impedance_sources={3: 0.05},
        )

        coeffs = _build_neumann_coefficients(
            SimpleNamespace(global_dof_count=2),
            tags,
            omega,
            config,
            np.complex128,
            excluded_tags=config.impedance_sources,
        )

        expected = 1j * config.air_density * omega
        np.testing.assert_allclose(coeffs, [expected, 0.0], rtol=1e-12)

    def test_axial_flat_disc_matches_normal(self):
        """A flat disc under axial reproduces the uniform-normal coefficients."""
        from hornlab_bempp_bem.bie import _build_neumann_data

        grid = _two_face_cap_grid(0.0)
        tags = np.array([2, 2], dtype=np.int32)
        omega = 2 * np.pi * 1000.0

        normal_gf, normal_cap = _captured_coefficients()
        with patch("bempp_cl.api.GridFunction", normal_gf):
            _build_neumann_data(
                SimpleNamespace(global_dof_count=2), tags, omega,
                SolveConfig(velocity_sources={2: 1.0}), "double",
            )
        axial_gf, axial_cap = _captured_coefficients()
        with patch("bempp_cl.api.GridFunction", axial_gf):
            _build_neumann_data(
                SimpleNamespace(global_dof_count=2), tags, omega,
                SolveConfig(velocity_sources={2: 1.0},
                            source_motion=SourceMotion.AXIAL),
                "double", grid=grid, source_axis=_FRAME_AXIS,
            )
        np.testing.assert_allclose(
            axial_cap["coefficients"], normal_cap["coefficients"], rtol=1e-12
        )

    def test_axial_curved_cap_tapers_rim(self):
        """A tilted/curved cap under axial scales each face by cos(theta)."""
        from hornlab_bempp_bem.bie import _build_neumann_data

        theta = np.deg2rad(40.0)
        grid = _two_face_cap_grid(theta)
        tags = np.array([2, 2], dtype=np.int32)
        omega = 2 * np.pi * 1500.0

        fake_gf, cap = _captured_coefficients()
        with patch("bempp_cl.api.GridFunction", fake_gf):
            _build_neumann_data(
                SimpleNamespace(global_dof_count=2), tags, omega,
                SolveConfig(velocity_sources={2: 1.0},
                            velocity_mode=VelocityMode.VELOCITY,
                            source_motion=SourceMotion.AXIAL),
                "double", grid=grid, source_axis=_FRAME_AXIS,
            )
        expected = 1j * 1.2041 * omega * 1.0 * np.cos(theta)
        np.testing.assert_allclose(cap["coefficients"], [expected, expected], rtol=1e-6)
