"""Per-source axial axes (``SolveConfig.source_axes``): contract tests 1-10.

The same numbers are asserted by hornlab-metal-bem so both solvers agree.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_bempp_bem.bie import _build_axial_element_scale
from hornlab_bempp_bem.config import (
    LinearSolver,
    ObservationConfig,
    SolveConfig,
    SourceMotion,
)

Z = np.array([0.0, 0.0, 1.0])
NZ = np.array([0.0, 0.0, -1.0])


def _grid(*normals):
    """One small triangle per requested outward normal (unit vectors)."""
    verts, elems = [], []
    for i, n in enumerate(normals):
        n = np.asarray(n, dtype=float)
        e1 = np.cross(n, [1.0, 0.0, 0.0] if abs(n[0]) < 0.9 else [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)  # e1 x e2 == n
        base = 3 * i
        verts += [np.zeros(3) + [3.0 * i, 0, 0], e1 + [3.0 * i, 0, 0],
                  e2 + [3.0 * i, 0, 0]]
        elems.append([base, base + 1, base + 2])
    return SimpleNamespace(
        vertices=np.array(verts).T, elements=np.array(elems, dtype=np.int32).T
    )


def _tilted(deg):
    t = np.deg2rad(deg)
    return [np.sin(t), 0.0, np.cos(t)]


def _explicit(grid, tags, axes, **kw):
    return _build_axial_element_scale(
        grid, np.array(tags, dtype=np.int32), list(axes), None,
        source_axes=axes, **kw,
    )


def _legacy(grid, tags, source_tags, axis):
    return _build_axial_element_scale(
        grid, np.array(tags, dtype=np.int32), source_tags, axis
    )


# 1 -----------------------------------------------------------------------
def test_flat_disc_sign_is_owned_by_the_caller():
    grid = _grid([0, 0, 1], [0, 0, 1])
    np.testing.assert_allclose(_explicit(grid, [2, 2], {2: Z}), [1.0, 1.0], atol=1e-15)
    np.testing.assert_allclose(_explicit(grid, [2, 2], {2: NZ}), [-1.0, -1.0], atol=1e-15)
    # Legacy pins the sign vote: frame -z still gives +1.
    np.testing.assert_allclose(_legacy(grid, [2, 2], [2], NZ), [1.0, 1.0], atol=1e-15)


# 2 -----------------------------------------------------------------------
def test_opposed_faces_and_tag_split_independence():
    grid = _grid([0, 0, 1], [0, 0, -1])
    np.testing.assert_allclose(_explicit(grid, [2, 2], {2: Z}), [1.0, -1.0], atol=1e-15)
    np.testing.assert_allclose(
        _explicit(grid, [2, 3], {2: Z, 3: Z}), [1.0, -1.0], atol=1e-15
    )
    # Legacy: each split tag votes its own sign, so the -z face flips to +1.
    np.testing.assert_allclose(_legacy(grid, [2, 3], [2, 3], Z), [1.0, 1.0], atol=1e-15)


# 3 -----------------------------------------------------------------------
def test_projection_cosine():
    grid = _grid(_tilted(60.0), _tilted(90.0))
    scale = _explicit(grid, [2, 3], {2: Z, 3: Z})
    assert abs(scale[0] - 0.5) < 1e-12
    assert abs(scale[1]) < 1e-12


# 4 -----------------------------------------------------------------------
def test_axis_is_normalized():
    grid = _grid(_tilted(60.0))
    np.testing.assert_array_equal(
        _explicit(grid, [2], {2: (0, 0, 2)}), _explicit(grid, [2], {2: (0, 0, 1)})
    )


# 5 -----------------------------------------------------------------------
@pytest.mark.parametrize("bad", [(0, 0, 0), (np.nan, 0, 1), (np.inf, 0, 0), (1, 2)])
def test_degenerate_axis_raises_legacy_and_explicit(bad):
    grid = _grid([0, 0, 1])
    with pytest.raises(ValueError):
        _legacy(grid, [2], [2], np.array(bad, dtype=float))
    with pytest.raises(ValueError):
        _explicit(grid, [2], {2: bad})


def test_degenerate_legacy_axis_with_no_source_faces_keeps_old_behaviour():
    grid = _grid([0, 0, 1])
    assert _legacy(grid, [2], [99], np.zeros(3)) is None


# 6 -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        dict(source_motion=SourceMotion.NORMAL, source_axes={2: Z}),  # not axial
        dict(source_motion=SourceMotion.AXIAL, source_axes={2: Z, 9: Z}),  # stray tag
        dict(source_motion=SourceMotion.AXIAL, source_axes={2: Z}),  # missing tag 3
        dict(source_motion=SourceMotion.AXIAL, source_axes={2: Z, 3: (0, 0, 0)}),
        dict(source_motion=SourceMotion.AXIAL, source_axes={2: Z, 3: (np.nan, 0, 1)}),
        dict(source_motion=SourceMotion.AXIAL, source_axes={2: Z, 3: (0, 1)}),
        dict(source_motion=SourceMotion.AXIAL, source_axes={2: Z, 3: (0, 1e-13, 0)}),
    ],
)
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        SolveConfig(velocity_sources={2: 1.0, 3: 1.0}, **kwargs)


def test_config_accepts_and_normalizes_valid_axes():
    cfg = SolveConfig(
        velocity_sources={2: 1.0, 3: 1.0},
        source_motion=SourceMotion.AXIAL,
        source_axes={2: (0, 0, 2), 3: [1, 0, 0]},
    )
    assert cfg.source_axes == {2: (0.0, 0.0, 2.0), 3: (1.0, 0.0, 0.0)}
    assert SolveConfig().source_axes is None


# 7 -----------------------------------------------------------------------
# This package supports native symmetry (planes "yz" x-normal, "xz" y-normal,
# "xy" z-normal, "yz+xz"), so the subspace check is live.
def test_symmetry_subspace_check():
    grid = _grid([0, 0, 1])
    with pytest.raises(ValueError, match="symmetry subspace"):
        _explicit(grid, [2], {2: (1, 0, 1)}, native_symmetry_plane="yz")
    with pytest.raises(ValueError, match="symmetry subspace"):
        _explicit(grid, [2], {2: (0, 1, 1)}, native_symmetry_plane="yz+xz")
    ok = _explicit(grid, [2], {2: (0, 0, 1)}, native_symmetry_plane="yz+xz")
    np.testing.assert_allclose(ok, [1.0], atol=1e-15)
    # Full model: any axis is accepted.
    _explicit(grid, [2], {2: (1, 0, 1)}, native_symmetry_plane=None)


# 8-10: solve level ---------------------------------------------------------
_ASSEMBLY_BACKEND = "opencl"


def _require_bempp_cpu():
    global _ASSEMBLY_BACKEND
    try:
        import bempp_cl.api  # noqa: F401
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"bempp-cl unavailable: {exc}")
    try:
        from hornlab_bempp_bem import configure_opencl

        configure_opencl("cpu")
        _ASSEMBLY_BACKEND = "opencl"
    except Exception:  # pragma: no cover
        _ASSEMBLY_BACKEND = "numba"


def _capped_sphere():
    """Sphere of radius 0.1 m; the +z cap is tag 2, the rest rigid tag 1."""
    import bempp_cl.api as bempp_api

    from hornlab_bempp_bem.mesh import LoadedMesh
    from hornlab_bempp_bem.result import MeshInfo

    grid = bempp_api.shapes.regular_sphere(2)
    verts = np.asarray(grid.vertices, dtype=float).T * 0.1
    tris = np.asarray(grid.elements, dtype=np.int64).T
    centroid_z = verts[tris].mean(axis=1)[:, 2]
    tags = np.where(centroid_z > 0.05, 2, 1).astype(np.int32)
    grid = bempp_api.Grid(
        np.ascontiguousarray(verts.T), np.ascontiguousarray(tris.T.astype(np.uint32))
    )
    return LoadedMesh(
        grid=grid,
        physical_tags=tags,
        info=MeshInfo(
            n_vertices=verts.shape[0],
            n_triangles=tris.shape[0],
            physical_groups={1: "wall", 2: "source"},
            bounding_box_m=(verts.min(axis=0), verts.max(axis=0)),
        ),
    )


def _frame(axis):
    from hornlab_bempp_bem.observation import ObservationFrame

    axis = np.asarray(axis, dtype=float)
    u = np.cross(axis, [0.3, 0.5, 0.8])
    u /= np.linalg.norm(u)
    return ObservationFrame(
        axis=axis, origin=np.zeros(3), u=u, v=np.cross(axis, u),
        mouth_center=np.zeros(3), source_center=np.zeros(3),
    )


def _solve(mesh, frame_axis=Z, **overrides):
    from hornlab_bempp_bem.sweep import run_sweep_serial

    points = np.array([[0.0, 0.0, 1.0], [0.7, 0.2, 0.6], [-0.5, 0.9, -0.3]])
    config = SolveConfig(
        velocity_sources={2: 1.0},
        source_motion=SourceMotion.AXIAL,
        solver=LinearSolver.LU,
        precision="double",
        assembly_backend=_ASSEMBLY_BACKEND,
        return_surface_traces=True,
        observation=ObservationConfig(
            planes=["horizontal"],
            custom_points={"horizontal": points},
            angle_count=points.shape[0],
        ),
        **overrides,
    )
    return run_sweep_serial(
        mesh, np.array([400.0]), _frame(frame_axis), config,
    )


@pytest.mark.slow
def test_polarity_flips_the_field():
    _require_bempp_cpu()
    mesh = _capped_sphere()
    up = _solve(mesh, source_axes={2: Z})
    down = _solve(mesh, source_axes={2: NZ})
    np.testing.assert_allclose(down.pressure_complex, -up.pressure_complex, rtol=1e-10)
    np.testing.assert_allclose(
        down.surface_pressure_complex, -up.surface_pressure_complex,
        rtol=1e-10, atol=1e-14,
    )
    assert np.abs(up.pressure_complex).max() > 0.0


@pytest.mark.slow
def test_explicit_up_axis_equals_legacy_bit_for_bit():
    _require_bempp_cpu()
    mesh = _capped_sphere()
    explicit = _solve(mesh, source_axes={2: Z})
    legacy = _solve(mesh, frame_axis=Z)
    np.testing.assert_array_equal(explicit.pressure_complex, legacy.pressure_complex)
    np.testing.assert_array_equal(
        explicit.surface_pressure_complex, legacy.surface_pressure_complex
    )
    np.testing.assert_array_equal(explicit.impedance, legacy.impedance)


@pytest.mark.slow
def test_observation_frame_does_not_change_the_explicit_drive():
    _require_bempp_cpu()
    mesh = _capped_sphere()
    a = _solve(mesh, frame_axis=Z, source_axes={2: (0.2, 0.0, 1.0)})
    b = _solve(mesh, frame_axis=[1.0, 0.0, 0.0], source_axes={2: (0.2, 0.0, 1.0)})
    np.testing.assert_array_equal(a.surface_pressure_complex, b.surface_pressure_complex)
    np.testing.assert_array_equal(a.pressure_complex, b.pressure_complex)
