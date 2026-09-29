from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.special import j1, struve

import hornlab_bempp_bem as bempp_bem
from hornlab_bempp_bem._constants import SPEED_OF_SOUND
from hornlab_bempp_bem.config import ObservationConfig, SolveConfig, VelocityMode
from hornlab_bempp_bem import infinite_baffle
from hornlab_bempp_bem.infinite_baffle import _validate_coupled_infinite_baffle
from hornlab_bempp_bem.mesh import LoadedMesh, MeshError, _resolve_coupled_ib_aperture_tag
from hornlab_bempp_bem.observation import ObservationFrame
from hornlab_bempp_bem.result import MeshInfo

TAG_THROAT = 2
TAG_WALL = 3
TAG_APERTURE = 12


def _triangulated_disc(
    radius: float,
    *,
    rings: int,
    sectors: int,
    z: float,
    normal_sign: int,
) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, z)]
    ring_indices: list[list[int]] = []
    for ring in range(1, rings + 1):
        row = []
        ring_radius = radius * ring / rings
        for sector in range(sectors):
            theta = 2.0 * np.pi * sector / sectors
            row.append(len(vertices))
            vertices.append(
                (
                    ring_radius * np.cos(theta),
                    ring_radius * np.sin(theta),
                    z,
                )
            )
        ring_indices.append(row)

    triangles: list[list[int]] = []
    first = ring_indices[0]
    for sector in range(sectors):
        nxt = (sector + 1) % sectors
        tri = [0, first[sector], first[nxt]]
        triangles.append(tri if normal_sign > 0 else [tri[0], tri[2], tri[1]])
    for ring in range(1, rings):
        inner = ring_indices[ring - 1]
        outer = ring_indices[ring]
        for sector in range(sectors):
            nxt = (sector + 1) % sectors
            pair = [
                [inner[sector], outer[sector], outer[nxt]],
                [inner[sector], outer[nxt], inner[nxt]],
            ]
            if normal_sign < 0:
                pair = [[a, c, b] for a, b, c in pair]
            triangles.extend(pair)
    return np.asarray(vertices), np.asarray(triangles, dtype=np.int32)


def _channel_mesh(
    radius: float = 0.04,
    depth: float = 0.004,
    *,
    rings: int = 2,
    sectors: int = 12,
    wall_layers: int = 1,
) -> LoadedMesh:
    """Straight circular channel; ``wall_layers`` element bands along the wall."""
    import bempp_cl.api as bempp_api

    top_vertices, top_triangles = _triangulated_disc(
        radius,
        rings=rings,
        sectors=sectors,
        z=0.0,
        normal_sign=1,
    )
    bottom_vertices, bottom_triangles = _triangulated_disc(
        radius,
        rings=rings,
        sectors=sectors,
        z=-depth,
        normal_sign=-1,
    )
    bottom_offset = top_vertices.shape[0]
    theta = 2.0 * np.pi * np.arange(sectors) / sectors
    middle_rings = [
        np.column_stack(
            [
                radius * np.cos(theta),
                radius * np.sin(theta),
                np.full(sectors, -depth + depth * layer / wall_layers),
            ]
        )
        for layer in range(1, wall_layers)
    ]
    vertices = np.vstack([top_vertices, bottom_vertices, *middle_rings])
    triangles = [*top_triangles.tolist()]
    tags = [TAG_APERTURE] * top_triangles.shape[0]
    triangles.extend((bottom_triangles + bottom_offset).tolist())
    tags.extend([TAG_THROAT] * bottom_triangles.shape[0])

    top_outer = 1 + (rings - 1) * sectors
    bottom_outer = bottom_offset + top_outer
    middle_start = bottom_offset + bottom_vertices.shape[0]
    # Outer-ring vertex indices from the throat (z=-depth) up to the mouth (z=0).
    ring_indices = [bottom_outer + np.arange(sectors)]
    ring_indices.extend(
        middle_start + layer * sectors + np.arange(sectors)
        for layer in range(wall_layers - 1)
    )
    ring_indices.append(top_outer + np.arange(sectors))
    for layer in range(wall_layers):
        lower, upper = ring_indices[layer], ring_indices[layer + 1]
        for sector in range(sectors):
            nxt = (sector + 1) % sectors
            triangles.extend(
                (
                    [lower[sector], lower[nxt], upper[nxt]],
                    [lower[sector], upper[nxt], upper[sector]],
                )
            )
            tags.extend((TAG_WALL, TAG_WALL))

    # Reverse the closed shell so every normal points into the acoustic cavity;
    # in particular, the aperture normal is -Z.
    triangles_array = np.asarray(triangles, dtype=np.int32)[:, [0, 2, 1]]
    tags_array = np.asarray(tags, dtype=np.int32)
    grid = bempp_api.Grid(vertices.T, triangles_array.T, tags_array)
    return LoadedMesh(
        grid=grid,
        physical_tags=tags_array,
        info=MeshInfo(
            n_vertices=vertices.shape[0],
            n_triangles=triangles_array.shape[0],
            physical_groups={
                TAG_THROAT: "throat",
                TAG_WALL: "wall",
                TAG_APERTURE: "mouth_aperture",
            },
            bounding_box_m=(vertices.min(axis=0), vertices.max(axis=0)),
        ),
        coupled_ib_aperture_tag=TAG_APERTURE,
    )


def _frame(depth: float = 0.004) -> ObservationFrame:
    origin = np.zeros(3, dtype=np.float64)
    return ObservationFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=origin,
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        mouth_center=origin,
        source_center=np.array([0.0, 0.0, -depth]),
    )


def _config(**overrides) -> SolveConfig:
    values = dict(
        aperture_tag=TAG_APERTURE,
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=VelocityMode.VELOCITY,
        frame_override=_frame(),
        assembly_backend="numba",
        precision="double",
        workers=1,
        observation=ObservationConfig(
            planes=["horizontal"],
            distance_m=1.5,
            angle_min_deg=0.0,
            angle_max_deg=180.0,
            angle_count=7,
        ),
    )
    values.update(overrides)
    return SolveConfig(**values)


def test_canonical_mouth_aperture_is_detected_without_numeric_tag_inference():
    tags = np.array([1, 2, 12], dtype=np.int32)
    assert _resolve_coupled_ib_aperture_tag(
        {12: "mouth_aperture"}, tags, None
    ) == 12
    assert _resolve_coupled_ib_aperture_tag({}, tags, None) is None
    with pytest.raises(MeshError, match="conflicts"):
        _resolve_coupled_ib_aperture_tag({12: "mouth_aperture"}, tags, 2)


def test_aperture_tag_validation_rejects_invalid_values():
    for value in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="aperture_tag"):
            SolveConfig(aperture_tag=value)


def test_coupled_ib_geometry_accepts_canonical_channel_and_rejects_wrong_frame():
    mesh = _channel_mesh()
    geometry = _validate_coupled_infinite_baffle(mesh, _config(), _frame())
    np.testing.assert_allclose(geometry.inward_normal, [0.0, 0.0, -1.0], atol=1e-12)
    np.testing.assert_allclose(geometry.outward_normal, [0.0, 0.0, 1.0], atol=1e-12)

    wrong_frame = _frame()
    wrong_frame.axis = -wrong_frame.axis
    with pytest.raises(ValueError, match="frame axis"):
        _validate_coupled_infinite_baffle(mesh, _config(), wrong_frame)


def test_coupled_ib_impedance_uses_lowest_driven_tag_and_zero_fallback(monkeypatch):
    bempp_api = pytest.importorskip("bempp_cl.api")
    from scipy.sparse import csr_matrix

    class MatrixOperator:
        def __init__(self, matrix):
            self.matrix = np.asarray(matrix, dtype=np.complex128)

        def __rmul__(self, value):
            return MatrixOperator(value * self.matrix)

        def __sub__(self, other):
            return MatrixOperator(self.matrix - other.matrix)

        def weak_form(self):
            return self

    geometry = infinite_baffle._ApertureGeometry(
        element_indices=np.array([2], dtype=np.int64),
        center=np.zeros(3),
        inward_normal=np.array([0.0, 0.0, -1.0]),
        outward_normal=np.array([0.0, 0.0, 1.0]),
    )
    monkeypatch.setattr(
        infinite_baffle, "_validate_coupled_infinite_baffle", lambda *_: geometry
    )
    monkeypatch.setattr(
        infinite_baffle,
        "build_observation_points",
        lambda *_: (np.array([[[0.0, 0.0, -1.0]]]), np.array([0.0])),
    )
    monkeypatch.setattr(
        infinite_baffle,
        "resolve_assembly_backend",
        lambda *_: SimpleNamespace(
            requested_backend="numba",
            effective_backend="numba",
            fallback_used=False,
            reason=None,
        ),
    )
    monkeypatch.setattr(infinite_baffle, "_operator_kwargs", lambda *_a, **_k: {})
    monkeypatch.setattr(
        infinite_baffle,
        "_build_p1_to_dp0_projection",
        lambda *_: csr_matrix(np.ones((3, 1))),
    )
    monkeypatch.setattr(
        infinite_baffle,
        "_build_neumann_data",
        lambda *_a, **_k: SimpleNamespace(coefficients=np.array([1.0 + 0.0j])),
    )
    monkeypatch.setattr(
        infinite_baffle,
        "_restrict_neumann_to_nonzero_support",
        lambda _grid, data: (SimpleNamespace(), data),
    )
    monkeypatch.setattr(
        infinite_baffle,
        "_evaluate_rayleigh_aperture",
        lambda *_: np.array([1.0 + 0.0j]),
    )
    monkeypatch.setattr(
        infinite_baffle,
        "_normalized_spl_db",
        lambda pressure, _index: np.zeros(pressure.shape, dtype=np.float64),
    )
    monkeypatch.setattr(
        infinite_baffle,
        "compute_surface_pressure_avg",
        lambda *_: {2: 1.0 + 0.0j, 3: 2.0 + 0.0j},
    )
    monkeypatch.setattr(
        bempp_api,
        "GridFunction",
        lambda _space, coefficients: SimpleNamespace(coefficients=coefficients),
    )

    boundary = bempp_api.operators.boundary
    monkeypatch.setattr(
        boundary.sparse,
        "identity",
        lambda *_a, **_k: MatrixOperator([[1.0]]),
    )
    monkeypatch.setattr(
        boundary.helmholtz,
        "double_layer",
        lambda *_a, **_k: MatrixOperator([[1.5]]),
    )

    mesh = SimpleNamespace(
        grid=SimpleNamespace(volumes=np.ones(3, dtype=np.float64)),
        physical_tags=np.array([2, 3, TAG_APERTURE], dtype=np.int32),
        info=MeshInfo(3, 3, {}, (np.zeros(3), np.ones(3))),
    )

    for sources, expected_tag in (
        ({2: 0.0, 3: 1.0}, 3),
        ({2: 0.0, 3: 0.0}, 2),
    ):
        p1_space = SimpleNamespace(global_dof_count=1)
        dp0_space = SimpleNamespace(global_dof_count=3)
        aperture_space = SimpleNamespace(global_dof_count=1)
        spaces = iter((p1_space, dp0_space, aperture_space))
        monkeypatch.setattr(
            bempp_api, "function_space", lambda *_a, **_k: next(spaces)
        )
        single_layer_matrices = iter(([[0.0]], [[1.0]], [[1.0]]))
        monkeypatch.setattr(
            boundary.helmholtz,
            "single_layer",
            lambda *_a, **_k: MatrixOperator(next(single_layer_matrices)),
        )
        monkeypatch.setattr(
            bempp_api,
            "as_matrix",
            lambda operator: operator.matrix,
        )

        result = infinite_baffle.run_coupled_infinite_baffle_sweep(
            mesh,
            np.array([1000.0]),
            _frame(),
            _config(velocity_sources=sources),
        )

        assert result.surface_pressure_avg is not None
        assert result.surface_pressure_avg[2][0] != result.surface_pressure_avg[3][0]
        assert result.impedance[0] == result.surface_pressure_avg[expected_tag][0]


@pytest.mark.slow
def test_bempp_coupled_ib_solves_forward_only_and_enforces_aperture_continuity():
    result = bempp_bem.solve_frequencies(_channel_mesh(), [1000.0], _config())

    assert result.frequencies_hz.tolist() == [1000.0]
    assert result.pressure_complex.shape == (1, 1, 7)
    assert np.abs(result.pressure_complex[0, 0, 0]) > 0.0
    # 180 degrees is behind the ideal baffle and must be exactly silent.
    assert result.pressure_complex[0, 0, -1] == 0.0
    # The 4 mm channel is acoustically shallow, so its aperture velocity is
    # nearly uniform and the forward hemisphere must follow the analytic Airy
    # pattern of a baffled circular piston.
    forward_angles = result.observation_angles_deg[:4]
    x = (
        2.0
        * np.pi
        * 1000.0
        / 343.0
        * 0.04
        * np.sin(np.deg2rad(forward_angles))
    )
    airy = np.ones_like(x)
    airy[1:] = 2.0 * j1(x[1:]) / x[1:]
    airy_db = 20.0 * np.log10(np.abs(airy))
    np.testing.assert_allclose(result.spl_db[0, 0, :4], airy_db, atol=0.05)
    diagnostics = result.solver_log[0]["native_diagnostics"]
    assert result.solver_log[0]["requested_solver"] == "gmres"
    assert result.solver_log[0]["effective_solver"] == "lu"
    assert result.solver_log[0]["requested_backend"] == "numba"
    assert result.solver_log[0]["effective_backend"] == "numba"
    assert result.solver_log[0]["fallback_used"] is False
    assert diagnostics["coupled_ib"] is True
    assert diagnostics["field"] == "rayleigh_aperture_only"
    assert diagnostics["aperture_velocity_basis"] == "DP0"
    assert diagnostics["requested_backend"] == "numba"
    assert diagnostics["effective_backend"] == "numba"
    assert diagnostics["aperture_pressure_continuity_rel"] < 1.0e-10


@pytest.mark.slow
def test_bempp_coupled_ib_matches_baffled_piston_absolute_field():
    """Pin the coupled field to the analytic shallow-channel piston limit."""
    frequency = 1000.0
    radius = 0.04
    distance = 1.5
    result = bempp_bem.solve_frequencies(_channel_mesh(), [frequency], _config())
    medium = _config()
    k = 2.0 * np.pi * frequency / SPEED_OF_SOUND
    # With exp(-i omega t), p = -i omega rho times the Rayleigh half-space
    # single layer. Its on-axis disc integral is (exp(ik R)-exp(ik d))/(ik).
    expected = -medium.air_density * SPEED_OF_SOUND * (
        np.exp(1j * k * np.hypot(distance, radius)) - np.exp(1j * k * distance)
    )
    measured = result.pressure_complex[0, 0, 0]
    assert abs(measured) / abs(expected) == pytest.approx(1.0, rel=0.15)
    assert abs(np.rad2deg(np.angle(measured / expected))) < 8.0

    angles = result.observation_angles_deg[:4]
    x = k * radius * np.sin(np.deg2rad(angles))
    airy = np.ones_like(x)
    airy[1:] = 2.0 * j1(x[1:]) / x[1:]
    np.testing.assert_allclose(
        result.spl_db[0, 0, :4], 20.0 * np.log10(np.abs(airy)), atol=0.05
    )
    assert result.pressure_complex[0, 0, -1] == 0.0


@pytest.mark.slow
@pytest.mark.parametrize("frequency", [800.0, 1600.0])
def test_bempp_coupled_ib_impedance_matches_baffled_piston(frequency: float):
    radius = 0.04
    result = bempp_bem.solve_frequencies(_channel_mesh(), [frequency], _config())
    medium = _config()
    ka = 2.0 * np.pi * frequency * radius / SPEED_OF_SOUND
    resistance = medium.air_density * SPEED_OF_SOUND * (1.0 - j1(2.0 * ka) / ka)
    # Negative reactance follows this solver's exp(-i omega t) convention.
    reactance = -medium.air_density * SPEED_OF_SOUND * struve(1, 2.0 * ka) / ka
    # Source pressure includes the 4 mm channel and coarse mesh bias.
    assert result.impedance[0].real == pytest.approx(resistance, rel=0.30)
    assert result.impedance[0].imag == pytest.approx(reactance, rel=0.30)


def _metal_engine_or_skip():
    """Return the Metal engine module, or skip when it cannot run here.

    hornlab-metal-bem is not a dependency of this package. The test needs it
    importable and its Swift/Metal helper runnable (Apple Silicon with a built
    helper); on every other host it skips.
    """
    metal_bem = pytest.importorskip("hornlab_metal_bem")
    from hornlab_metal_bem.metal import discover_native_runtime

    status = discover_native_runtime(run_smoke_test=True)
    if not status.available:
        pytest.skip(
            "Swift/Metal native helper unavailable: "
            + "; ".join(status.unavailable_reasons)
        )
    return metal_bem


@pytest.mark.slow
def test_bempp_and_metal_coupled_ib_agree_on_the_same_resonant_channel():
    """Two independent engines, one mesh: the resonant deep channel must agree.

    Fixture: 40 mm radius, 100 mm deep straight channel, 6 rings x 36 sectors and
    15 wall layers (1872 triangles), unit throat velocity, on-axis and off-axis
    points 1.5 m from the mouth, at and around the first resonance (about 650 Hz)
    and up to 2 kHz. Both engines use the standard formulation (real k) in double
    precision. Measured agreement on this mesh: 0.0002 dB and 0.0006 deg against
    the current Metal helper (0.0005 dB / 0.003 deg against the older installed
    0.1.0 helper; the earlier probe saw 0.01 dB / 0.1 deg with a coarser
    frequency set). The limits below (0.02 dB, 0.3 deg) leave two orders of
    margin for GPU float32 noise on other machines, and still fail on a 7.5 %
    aperture-coupling error in Metal (measured 0.74 dB / 15 deg) or a halved one
    (6.1 dB / 80 deg). They are far below the 0.6 dB / 7 deg gap either engine
    has against the 1-D pipe reference (see hornlab-metal-bem
    tests/test_native_coupled_ib_validation.py), so that gap does not hide here.
    """
    metal_bem = _metal_engine_or_skip()
    from hornlab_metal_bem.config import ObservationConfig as MetalObservation
    from hornlab_metal_bem.config import SolveConfig as MetalSolveConfig
    from hornlab_metal_bem.config import VelocityMode as MetalVelocityMode
    from hornlab_metal_bem.mesh import LoadedMesh as MetalMesh
    from hornlab_metal_bem.mesh import make_pure_grid
    from hornlab_metal_bem.observation import ObservationFrame as MetalFrame
    from hornlab_metal_bem.result import MeshInfo as MetalMeshInfo

    depth = 0.10
    frequencies = [300.0, 640.0, 650.0, 659.0, 670.0, 1000.0, 1500.0, 2000.0]
    bempp_mesh = _channel_mesh(0.04, depth, rings=6, sectors=36, wall_layers=15)
    vertices = np.asarray(bempp_mesh.grid.vertices, dtype=np.float64).T.copy()
    elements = np.asarray(bempp_mesh.grid.elements, dtype=np.int32).T.copy()
    metal_mesh = MetalMesh(
        grid=make_pure_grid(vertices, elements),
        physical_tags=np.asarray(bempp_mesh.physical_tags, dtype=np.int32),
        info=MetalMeshInfo(
            n_vertices=vertices.shape[0],
            n_triangles=elements.shape[0],
            physical_groups={
                TAG_THROAT: "throat",
                TAG_WALL: "wall",
                TAG_APERTURE: "aperture",
            },
            bounding_box_m=(vertices.min(axis=0), vertices.max(axis=0)),
        ),
    )
    origin = np.zeros(3, dtype=np.float64)
    metal_frame = MetalFrame(
        axis=np.array([0.0, 0.0, 1.0]),
        origin=origin,
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        mouth_center=origin,
        source_center=np.array([0.0, 0.0, -depth]),
    )
    metal_config = MetalSolveConfig(
        velocity_sources={TAG_THROAT: 1.0},
        velocity_mode=MetalVelocityMode.VELOCITY,
        aperture_tag=TAG_APERTURE,
        observation=MetalObservation(
            distance_m=1.5,
            angle_min_deg=0.0,
            angle_max_deg=90.0,
            angle_count=4,
            planes=["horizontal"],
            origin="mouth",
        ),
        frame_override=metal_frame,
        formulation="standard",
        metal_native_assembly_mode="corrected",
        dense_solve_dtype="float64",
    )
    bempp_config = _config(
        frame_override=_frame(depth),
        formulation="standard",
        observation=ObservationConfig(
            planes=["horizontal"],
            distance_m=1.5,
            angle_min_deg=0.0,
            angle_max_deg=90.0,
            angle_count=4,
        ),
    )

    metal_result = metal_bem.solve_frequencies(metal_mesh, frequencies, metal_config)
    bempp_result = bempp_bem.solve_frequencies(bempp_mesh, frequencies, bempp_config)

    assert all(e.get("coupled_ib") is True for e in metal_result.native_diagnostics)
    assert all(
        entry["native_diagnostics"]["coupled_ib"] is True
        for entry in bempp_result.solver_log
    )
    metal_p = metal_result.pressure_complex
    bempp_p = bempp_result.pressure_complex
    assert metal_p.shape == bempp_p.shape == (len(frequencies), 1, 4)
    ratio = bempp_p / metal_p
    level_db = np.abs(20.0 * np.log10(np.abs(ratio)))
    phase_deg = np.abs(np.degrees(np.angle(ratio)))
    assert float(level_db.max()) < 0.02, level_db
    assert float(phase_deg.max()) < 0.3, phase_deg
