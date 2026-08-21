from __future__ import annotations

import numpy as np
import pytest
from scipy.special import j1

import hornlab_bempp_bem as bempp_bem
from hornlab_bempp_bem.config import ObservationConfig, SolveConfig, VelocityMode
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
) -> LoadedMesh:
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
    vertices = np.vstack([top_vertices, bottom_vertices])
    bottom_offset = top_vertices.shape[0]
    triangles = [*top_triangles.tolist()]
    tags = [TAG_APERTURE] * top_triangles.shape[0]
    triangles.extend((bottom_triangles + bottom_offset).tolist())
    tags.extend([TAG_THROAT] * bottom_triangles.shape[0])

    top_outer = 1 + (rings - 1) * sectors
    bottom_outer = bottom_offset + top_outer
    for sector in range(sectors):
        nxt = (sector + 1) % sectors
        top0, top1 = top_outer + sector, top_outer + nxt
        bottom0, bottom1 = bottom_outer + sector, bottom_outer + nxt
        triangles.extend(
            ([bottom0, bottom1, top1], [bottom0, top1, top0])
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
    assert diagnostics["coupled_ib"] is True
    assert diagnostics["field"] == "rayleigh_aperture_only"
    assert diagnostics["aperture_velocity_basis"] == "DP0"
    assert diagnostics["aperture_pressure_continuity_rel"] < 1.0e-10
