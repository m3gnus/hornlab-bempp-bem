"""Rigid half-space ground plane.

The physics claim under test is that ``ground_plane`` solves the same problem
as the body plus its physical mirror image radiating into free space. That
reference is computed on the ordinary non-symmetric code path, which shares no
machinery with the image assembler, so agreement is evidence rather than a
tautology.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from hornlab_bempp_bem.config import (
    BIEFormulation,
    LinearSolver,
    ObservationConfig,
    SolveConfig,
    reject_unsupported_native_symmetry,
    uses_image_assembly,
)
from hornlab_bempp_bem.symmetry import _image_signs, expand_symmetry_mesh


# ---------------------------------------------------------------------------
# Image group
# ---------------------------------------------------------------------------

def test_existing_symmetry_specs_keep_their_exact_image_order():
    """Generalising the parser must not renumber the existing image blocks.

    Element blocks are stacked in this order and the reduced solve indexes the
    first one as the real body, so a reordering here would silently mirror the
    wrong block.
    """
    assert _image_signs("yz") == ((1, 1, 1), (-1, 1, 1))
    assert _image_signs("xz") == ((1, 1, 1), (1, -1, 1))
    assert _image_signs("yz+xz") == (
        (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1),
    )


def test_ground_plane_reflections_match_beat():
    """BEAT's ``rigid_ground_transform`` is signs (1, -1, 1), determinant -1."""
    assert _image_signs("xz") == ((1, 1, 1), (1, -1, 1))
    for signs in _image_signs("xz")[1:]:
        assert int(np.prod(signs)) == -1
    # The z mirror is reachable only as a ground plane.
    assert _image_signs("xy") == ((1, 1, 1), (1, 1, -1))


def test_plane_spec_is_order_independent_and_rejects_nonsense():
    assert _image_signs("xz+yz") == _image_signs("yz+xz")
    with pytest.raises(ValueError, match="unknown mirror plane"):
        _image_signs("zz")
    with pytest.raises(ValueError, match="repeats a plane"):
        _image_signs("yz+yz")


# ---------------------------------------------------------------------------
# Config surface
# ---------------------------------------------------------------------------

def test_ground_plane_defaults_off_and_validates():
    assert SolveConfig().ground_plane is None
    assert not uses_image_assembly(SolveConfig())
    for name in ("yz", "xz", "xy"):
        assert uses_image_assembly(SolveConfig(ground_plane=name))
    with pytest.raises(ValueError, match="ground_plane must be"):
        SolveConfig(ground_plane="zz")


def test_ground_plane_may_not_double_as_a_symmetry_plane():
    with pytest.raises(ValueError, match="cannot also be an infinite rigid"):
        SolveConfig(native_symmetry_plane="yz+xz", ground_plane="xz")
    # Distinct planes compose.
    config = SolveConfig(native_symmetry_plane="yz", ground_plane="xy")
    assert uses_image_assembly(config)


def test_ground_plane_inherits_the_image_assembler_limits():
    with pytest.raises(NotImplementedError, match="STANDARD and COMPLEX_K"):
        reject_unsupported_native_symmetry(
            SolveConfig(
                ground_plane="xy", formulation=BIEFormulation.BURTON_MILLER,
            )
        )
    with pytest.raises(NotImplementedError, match="Robin impedance"):
        reject_unsupported_native_symmetry(
            SolveConfig(ground_plane="xy", impedance_sources={1: 0.05}),
        )


# ---------------------------------------------------------------------------
# Geometry classification
# ---------------------------------------------------------------------------

def _open_box(z0: float, z1: float, drop_bottom: bool):
    """Axis-aligned box, optionally with its bottom face removed."""
    x0, x1, y0, y1 = -0.2, 0.2, -0.15, 0.15
    corners = np.array(
        [[x, y, z] for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)],
        dtype=np.float64,
    )
    quads = [
        (0, 1, 3, 2),  # bottom  z0
        (4, 6, 7, 5),  # top     z1
        (0, 4, 5, 1),  # -y
        (2, 3, 7, 6),  # +y
        (0, 2, 6, 4),  # -x
        (1, 5, 7, 3),  # +x
    ]
    if drop_bottom:
        quads = quads[1:]
    tris = []
    for a, b, c, d in quads:
        tris += [[a, b, c], [a, c, d]]
    tris = np.array(tris, dtype=np.int64)
    return corners, tris, np.full(tris.shape[0], 2, dtype=np.int32)


def test_body_clear_of_the_ground_mirrors_without_welding():
    verts, tris, tags = _open_box(0.30, 0.60, drop_bottom=False)
    expanded = expand_symmetry_mesh(verts, tris, tags, None, ground_plane="xy")
    assert expanded.ground_plane == "xy"
    assert expanded.seam_planes == ()          # detached: nothing to weld
    assert expanded.plane_spec == "xy"
    assert len(expanded.image_signs) == 2
    assert expanded.triangles_nx3.shape[0] == 2 * tris.shape[0]
    # Two disjoint closed shells, so no vertex is shared.
    assert expanded.vertices_nx3.shape[0] == 2 * verts.shape[0]


def test_body_resting_on_the_ground_welds_its_footprint():
    verts, tris, tags = _open_box(0.0, 0.35, drop_bottom=True)
    expanded = expand_symmetry_mesh(verts, tris, tags, None, ground_plane="xy")
    assert expanded.seam_planes == ("xy",)     # resting: the footprint is a cut
    # The four bottom corners are shared with the image.
    assert expanded.vertices_nx3.shape[0] == 2 * verts.shape[0] - 4


def test_a_closed_body_sitting_on_the_ground_is_rejected():
    """Its bottom face would coincide with its own image."""
    verts, tris, tags = _open_box(0.0, 0.35, drop_bottom=False)
    with pytest.raises(ValueError, match="no open boundary edge lying in it"):
        expand_symmetry_mesh(verts, tris, tags, None, ground_plane="xy")


def test_a_body_crossing_the_ground_is_rejected():
    verts, tris, tags = _open_box(-0.1, 0.35, drop_bottom=False)
    with pytest.raises(ValueError, match="crosses its .* ground plane"):
        expand_symmetry_mesh(verts, tris, tags, None, ground_plane="xy")


def test_a_symmetry_plane_may_not_also_be_the_ground_plane():
    verts, tris, tags = _open_box(0.30, 0.60, drop_bottom=False)
    with pytest.raises(ValueError, match="both a symmetry plane and the ground"):
        expand_symmetry_mesh(verts, tris, tags, "xy", ground_plane="xy")


def test_observation_below_the_ground_plane_warns(caplog):
    from hornlab_bempp_bem.sweep import _warn_observation_below_ground

    points = np.array(
        [[1.0, 0.0, 2.0], [1.0, 0.0, -0.5], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    config = SolveConfig(ground_plane="xy")
    with caplog.at_level(logging.WARNING, logger="hornlab_bempp_bem.sweep"):
        # A point exactly on the plane is on the domain boundary, not below it.
        assert _warn_observation_below_ground(config, points) == 1
    assert "inside the rigid half space" in caplog.text
    assert _warn_observation_below_ground(SolveConfig(), points) == 0


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

def _require_bempp_cpu():
    try:
        import bempp_cl.api  # noqa: F401
    except Exception as exc:  # pragma: no cover - depends on env
        pytest.skip(f"bempp-cl unavailable: {exc}")
    try:
        from hornlab_bempp_bem import configure_opencl

        configure_opencl("cpu")
    except Exception as exc:  # pragma: no cover - depends on env
        pytest.skip(f"OpenCL CPU runtime unavailable: {exc}")


_RADIUS = 0.15
_HEIGHT = 0.45
_FREQUENCY_HZ = 700.0


def _driven_sphere(component: int):
    """Unit sphere scaled and lifted clear of the plane, driven on every face."""
    import bempp_cl.api as bempp_api

    grid = bempp_api.shapes.regular_sphere(3)
    verts = np.asarray(grid.vertices, dtype=np.float64).T * _RADIUS
    verts[:, component] += _HEIGHT
    tris = np.asarray(grid.elements, dtype=np.int64).T
    return verts, tris, np.full(tris.shape[0], 2, dtype=np.int32)


def _loaded(verts, tris, tags):
    import bempp_cl.api as bempp_api

    from hornlab_bempp_bem.mesh import LoadedMesh
    from hornlab_bempp_bem.result import MeshInfo

    grid = bempp_api.Grid(
        np.ascontiguousarray(verts.T),
        np.ascontiguousarray(tris.T.astype(np.uint32)),
    )
    return LoadedMesh(
        grid=grid,
        physical_tags=np.asarray(tags, dtype=np.int32),
        info=MeshInfo(
            n_vertices=verts.shape[0],
            n_triangles=tris.shape[0],
            physical_groups={2: "source"},
            bounding_box_m=(verts.min(axis=0), verts.max(axis=0)),
        ),
    )


def _solve_at(mesh, points, **overrides):
    from hornlab_bempp_bem.observation import ObservationFrame
    from hornlab_bempp_bem.sweep import run_sweep_serial

    frame = ObservationFrame(
        axis=np.array([1.0, 0.0, 0.0]),
        origin=np.zeros(3),
        u=np.array([0.0, 1.0, 0.0]),
        v=np.array([0.0, 0.0, 1.0]),
        mouth_center=np.zeros(3),
        source_center=np.zeros(3),
    )
    config = SolveConfig(
        velocity_sources={2: 1.0},
        solver=LinearSolver.LU,
        precision="double",
        assembly_backend="opencl",
        observation=ObservationConfig(
            planes=["horizontal"],
            custom_points={"horizontal": points},
            angle_count=points.shape[0],
        ),
        **overrides,
    )
    result = run_sweep_serial(
        mesh, np.array([_FREQUENCY_HZ]), frame, config,
    )
    return result.pressure_complex[0, 0], result.impedance[0]


def _half_space_points(component: int, count: int, seed: int):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3))
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    points[:, component] = np.abs(points[:, component]) + 0.15
    return points * rng.uniform(1.3, 3.0, (count, 1))


@pytest.mark.slow
@pytest.mark.parametrize(
    "ground_plane,component", [("yz", 0), ("xz", 1), ("xy", 2)],
)
def test_ground_plane_equals_the_body_plus_its_image_in_free_space(
    ground_plane, component,
):
    """The decisive check, and the only one that pins the reflection sign.

    A rigid plane and a physically doubled body are the same problem. The
    reference solve runs on the plain non-symmetric path over a mesh that
    contains the mirrored body explicitly, with its winding reversed so the
    image's outward normal is the reflection of the original's. Any error in
    the image transform, the orbit tying, the Neumann extension, or the field
    evaluation shows up here as a phase or amplitude difference.
    """
    _require_bempp_cpu()

    verts, tris, tags = _driven_sphere(component)
    points = _half_space_points(component, 24, seed=17 + component)

    mirrored = verts.copy()
    mirrored[:, component] *= -1.0
    doubled_verts = np.vstack([verts, mirrored])
    doubled_tris = np.vstack([tris, tris[:, [0, 2, 1]] + verts.shape[0]])
    doubled_tags = np.concatenate([tags, tags])

    ground, ground_z = _solve_at(
        _loaded(verts, tris, tags), points, ground_plane=ground_plane,
    )
    reference, reference_z = _solve_at(
        _loaded(doubled_verts, doubled_tris, doubled_tags), points,
    )

    relative = np.abs(ground - reference) / np.abs(reference)
    assert relative.max() < 1.0e-6, relative.max()
    assert abs(ground_z - reference_z) / abs(reference_z) < 1.0e-6


@pytest.mark.slow
def test_a_rigid_plane_doubles_the_field_on_it_rather_than_cancelling_it():
    """Reflection coefficient +1, not -1.

    On the plane the direct and image contributions are exactly in phase, so a
    rigid boundary doubles the isolated body's field. A pressure-release plane
    -- the sign error this guards against -- would cancel it to zero.
    """
    _require_bempp_cpu()

    verts, tris, tags = _driven_sphere(2)
    on_plane = np.stack(
        [np.zeros(5), np.linspace(1.0, 2.5, 5), np.zeros(5)], axis=1,
    )

    with_ground, _ = _solve_at(
        _loaded(verts, tris, tags), on_plane, ground_plane="xy",
    )
    isolated, _ = _solve_at(_loaded(verts, tris, tags), on_plane)

    ratio = with_ground / isolated
    # The sphere is not a point source and its image sits 0.9 m away, so the
    # ratio carries a small finite-size correction; the sign is what matters.
    assert np.all(np.abs(ratio.real - 2.0) < 0.05), ratio
    assert np.all(np.abs(ratio) > 1.9), ratio


@pytest.mark.slow
def test_ground_plane_composes_with_a_reduced_quarter_mesh():
    """Cut on 'yz', standing above 'xy': four images on one reduced row block.

    The reference is the same expanded geometry solved without any image
    reduction, so this isolates the reduction from the ground physics that the
    two-body test already pinned.
    """
    _require_bempp_cpu()

    from hornlab_bempp_bem.symmetry import expand_symmetry_mesh

    import bempp_cl.api as bempp_api

    grid = bempp_api.shapes.regular_sphere(3)
    verts = np.asarray(grid.vertices, dtype=np.float64).T * 0.15
    tris = np.asarray(grid.elements, dtype=np.int64).T
    tags = np.full(tris.shape[0], 2, dtype=np.int32)

    # Keep the +x half of the sphere: the cut lands on the yz plane, so the
    # reduced mesh has a real open boundary there. Lift it clear of z = 0.
    centroids = verts[tris].mean(axis=1)
    keep = centroids[:, 0] > 0.0
    tris = tris[keep]
    tags = tags[keep]
    used, tris = np.unique(tris, return_inverse=True)
    tris = tris.reshape(-1, 3)
    verts = verts[used]
    verts[np.abs(verts[:, 0]) < 1.0e-9, 0] = 0.0
    verts[:, 2] += 0.40

    expanded = expand_symmetry_mesh(
        verts, tris, tags, "yz", ground_plane="xy",
    )
    assert len(expanded.image_signs) == 4
    assert expanded.seam_planes == ("yz",)

    points = _half_space_points(2, 20, seed=31)
    reduced, reduced_z = _solve_at(
        _loaded(verts, tris, tags), points,
        native_symmetry_plane="yz", ground_plane="xy",
    )
    full, full_z = _solve_at(
        _loaded(
            expanded.vertices_nx3,
            expanded.triangles_nx3,
            expanded.physical_tags,
        ),
        points,
    )

    relative = np.abs(reduced - full) / np.abs(full)
    # Looser than the two-body bound: the reduced P1 space keeps its cut-plane
    # boundary dofs while the expanded mesh's space does not, which is the same
    # seam-level difference the pure-symmetry parity tests already carry.
    assert relative.max() < 5.0e-3, relative.max()
    assert abs(reduced_z - full_z) / abs(full_z) < 5.0e-3
