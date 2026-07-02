"""Parity tests for ``hornlab_bempp_bem.observation.infer_frame``.

Exercises the cases WG's ``infer_observation_frame`` historically
handled "more robustly":

- enclosed waveguide (source disc sits in the middle of the mesh)
- freestanding horn (source at one extreme)
- BIGMEH-style cabinet with multiple source-tagged elements
- mixed source-element winding (sign-aligned normal sum)
- defensive handling of stale element indices
- symmetry-plane projection (yz / xy)

These are the behaviours downstream consumers used to import from WG's
legacy frame inference. They now live in canonical ``infer_frame``.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from hornlab_bempp_bem.observation import infer_frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_grid(vertices: np.ndarray, elements: np.ndarray):
    grid = MagicMock()
    grid.vertices = vertices  # (3, N)
    grid.elements = elements  # (3, M)
    grid.number_of_elements = elements.shape[1]
    return grid


def _build_curved_cap_horn_mesh():
    """Build a symmetric +Z horn with a curved source cap."""
    count = 64
    radial_count = 5
    throat_radius = 0.02
    mouth_radius = 0.08
    length = 0.3
    sag = 0.5 * throat_radius
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)

    center = np.array([[0.0, 0.0, 0.0]])
    rings = []
    ring_indices = []
    next_vertex = 1
    for ring_idx in range(1, radial_count + 1):
        radius = throat_radius * ring_idx / radial_count
        z = sag * (radius / throat_radius) ** 2
        ring = np.column_stack([
            radius * np.cos(angles),
            radius * np.sin(angles),
            z * np.ones(count),
        ])
        rings.append(ring)
        ring_indices.append(np.arange(next_vertex, next_vertex + count))
        next_vertex += count

    mouth = np.column_stack([
        mouth_radius * np.cos(angles),
        mouth_radius * np.sin(angles),
        length * np.ones(count),
    ])
    mouth_indices = np.arange(next_vertex, next_vertex + count)
    vertices = np.vstack([center, *rings, mouth])

    source_elems = []
    wall_elems = []
    for idx in range(count):
        next_idx = (idx + 1) % count
        source_elems.append([0, ring_indices[0][idx], ring_indices[0][next_idx]])
        for ring_idx in range(radial_count - 1):
            inner_idx = ring_indices[ring_idx][idx]
            inner_next = ring_indices[ring_idx][next_idx]
            outer_idx = ring_indices[ring_idx + 1][idx]
            outer_next = ring_indices[ring_idx + 1][next_idx]
            source_elems.append([inner_idx, outer_idx, outer_next])
            source_elems.append([inner_idx, outer_next, inner_next])

        rim_idx = ring_indices[-1][idx]
        rim_next = ring_indices[-1][next_idx]
        mouth_idx = mouth_indices[idx]
        mouth_next = mouth_indices[next_idx]
        wall_elems.append([rim_idx, mouth_idx, mouth_next])
        wall_elems.append([rim_idx, mouth_next, rim_next])

    elements = np.array(source_elems + wall_elems, dtype=np.int32)
    tags = np.array([2] * len(source_elems) + [1] * len(wall_elems), dtype=np.int32)
    return vertices.T, elements.T, tags


def _restrict_mesh(
    vertices: np.ndarray,
    elements: np.ndarray,
    tags: np.ndarray,
    predicate,
):
    """Keep elements whose centroid satisfies predicate and compact vertices."""
    vertices_nx3 = vertices.T
    elements_mx3 = elements.T
    centroids = vertices_nx3[elements_mx3].mean(axis=1)
    keep = np.array([predicate(centroid) for centroid in centroids], dtype=bool)

    kept_elements = elements_mx3[keep]
    kept_tags = tags[keep]
    used_vertices = np.unique(kept_elements)
    remap = np.full(vertices_nx3.shape[0], -1, dtype=np.int32)
    remap[used_vertices] = np.arange(used_vertices.shape[0], dtype=np.int32)

    return vertices_nx3[used_vertices].T, remap[kept_elements].T, kept_tags


def _rotate_z_horn_to_y(vertices: np.ndarray) -> np.ndarray:
    """Rotate +Z horn coordinates so it fires along +Y."""
    vertices_nx3 = vertices.T
    rotated = np.column_stack([
        vertices_nx3[:, 0],
        vertices_nx3[:, 2],
        -vertices_nx3[:, 1],
    ])
    return rotated.T


# ---------------------------------------------------------------------------
# Freestanding horn — source at one extreme of the span
# ---------------------------------------------------------------------------

def test_freestanding_horn_axis_points_from_throat_to_mouth():
    """Classic horn: source disc at z=0, mouth at z=1. Forward axis = +z."""
    # Source disc (4 verts, 2 triangles, normal +z)
    src_verts = np.array([
        [-0.02, -0.02, 0.0], [0.02, -0.02, 0.0],
        [0.02, 0.02, 0.0], [-0.02, 0.02, 0.0],
    ])
    # Horn body extending forward to z=1
    body_verts = np.array([
        [-0.2, -0.2, 1.0], [0.2, -0.2, 1.0],
        [0.2, 0.2, 1.0], [-0.2, 0.2, 1.0],
    ])
    vertices = np.vstack([src_verts, body_verts])
    src_elems = np.array([[0, 1, 2], [0, 2, 3]])
    body_elems = np.array([[4, 5, 6], [4, 6, 7]])
    elements = np.vstack([src_elems, body_elems])
    tags = np.array([2, 2, 1, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)
    frame = infer_frame(grid, tags, source_tag=2, origin_at="mouth")

    # Axis points strongly +z
    assert frame.axis[2] > 0.9
    # Mouth origin near z=1, source centre near z=0
    assert frame.origin[2] > 0.8
    assert abs(frame.source_center[2]) < 0.05


# ---------------------------------------------------------------------------
# Enclosed waveguide — source disc in the middle of the mesh
# ---------------------------------------------------------------------------

def test_enclosed_waveguide_trusts_source_normal_over_extent():
    """When ``enc_depth > 2 * horn_length`` the source sits ~midway
    between the horn mouth and the enclosure back wall. The extent
    heuristic would flip the axis; the canonical implementation must
    trust the source normal instead.
    """
    # Source disc at z=0.5 with normal +z
    src_verts = np.array([
        [-0.02, -0.02, 0.5], [0.02, -0.02, 0.5],
        [0.02, 0.02, 0.5], [-0.02, 0.02, 0.5],
    ])
    # Horn mouth ahead at z=0.7
    mouth_verts = np.array([
        [-0.2, -0.2, 0.7], [0.2, -0.2, 0.7],
        [0.2, 0.2, 0.7], [-0.2, 0.2, 0.7],
    ])
    # Enclosure back wall far behind at z=0.0
    back_verts = np.array([
        [-0.3, -0.3, 0.0], [0.3, -0.3, 0.0],
        [0.3, 0.3, 0.0], [-0.3, 0.3, 0.0],
    ])
    vertices = np.vstack([src_verts, mouth_verts, back_verts])

    src_elems = np.array([[0, 1, 2], [0, 2, 3]])
    mouth_elems = np.array([[4, 5, 6], [4, 6, 7]])
    back_elems = np.array([[8, 9, 10], [8, 10, 11]])
    elements = np.vstack([src_elems, mouth_elems, back_elems])
    tags = np.array([2, 2, 1, 1, 1, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)
    frame = infer_frame(grid, tags, source_tag=2, origin_at="mouth")

    # Source normal says +z; back wall at z=0 is further than mouth at z=0.7,
    # so the naive extent test (mouth_at_max) would point the axis at -z.
    # Canonical implementation must trust the normal.
    assert frame.axis[2] > 0.9


# ---------------------------------------------------------------------------
# BIGMEH cabinet — multiple source-tagged elements (multi-driver)
# ---------------------------------------------------------------------------

def test_bigmeh_cabinet_multiple_source_elements():
    """A BIGMEH-style cabinet may carry many source-tagged elements
    across multiple drivers. The area-weighted normal sum should still
    converge to the cabinet-forward axis.
    """
    # Two driver discs, both facing +y, at slightly different positions
    src_disc_1 = np.array([
        [-0.05, 0.0, -0.1], [0.05, 0.0, -0.1],
        [0.05, 0.0, 0.1], [-0.05, 0.0, 0.1],
    ])
    src_disc_2 = np.array([
        [-0.05, 0.0, -0.4], [0.05, 0.0, -0.4],
        [0.05, 0.0, -0.2], [-0.05, 0.0, -0.2],
    ])
    # Cabinet front face at y=0.5
    front_verts = np.array([
        [-0.3, 0.5, -0.5], [0.3, 0.5, -0.5],
        [0.3, 0.5, 0.5], [-0.3, 0.5, 0.5],
    ])
    vertices = np.vstack([src_disc_1, src_disc_2, front_verts])

    # Source tris (winding chosen so normals point +y)
    src_elems = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
    ])
    body_elems = np.array([[8, 9, 10], [8, 10, 11]])
    elements = np.vstack([src_elems, body_elems])
    tags = np.array([2, 2, 2, 2, 1, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)
    frame = infer_frame(grid, tags, source_tag=2, origin_at="mouth")

    # Axis should point +y toward the cabinet front
    assert frame.axis[1] > 0.9


# ---------------------------------------------------------------------------
# Mixed source winding — sign-aligned normal sum must not cancel
# ---------------------------------------------------------------------------

def test_mixed_winding_does_not_cancel_axis():
    """If some source triangles are wound CW and others CCW (legacy gmsh
    quirk), the raw sum of cross-products would partially cancel. WG's
    robust path sign-aligns normals into one hemisphere first.
    """
    # Four source tris at z=0; two with +z normal, two with -z normal
    base = np.array([
        [-0.01, -0.01, 0.0], [0.01, -0.01, 0.0],
        [0.01, 0.01, 0.0], [-0.01, 0.01, 0.0],
    ])
    body_vert = np.array([[0.0, 0.0, 0.5]])
    vertices = np.vstack([base, body_vert])

    # Tris 0-1: CCW winding → normal +z
    # Tris 2-3: CW winding → normal -z
    src_elems = np.array([
        [0, 1, 2], [0, 2, 3],   # +z
        [0, 2, 1], [0, 3, 2],   # -z
    ])
    body_elem = np.array([[0, 1, 4]])
    elements = np.vstack([src_elems, body_elem])
    tags = np.array([2, 2, 2, 2, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)
    frame = infer_frame(grid, tags, source_tag=2, origin_at="mouth")

    # With sign-alignment the axis magnitude must be well-defined;
    # source-at-min ⇒ axis +z; body vert pulls mouth to +z.
    axis_norm = float(np.linalg.norm(frame.axis))
    assert abs(axis_norm - 1.0) < 1e-6
    assert frame.axis[2] > 0.9


# ---------------------------------------------------------------------------
# Defensive: stale element indices
# ---------------------------------------------------------------------------

def test_stale_element_indices_are_skipped():
    """Legacy meshes occasionally carry element rows that index past the
    vertex array. The canonical implementation must drop these rows
    rather than IndexError.
    """
    src_verts = np.array([
        [-0.02, -0.02, 0.0], [0.02, -0.02, 0.0],
        [0.02, 0.02, 0.0], [-0.02, 0.02, 0.0],
    ])
    body_verts = np.array([
        [-0.2, -0.2, 1.0], [0.2, -0.2, 1.0],
    ])
    vertices = np.vstack([src_verts, body_verts])

    # First two source tris valid; third indexes a non-existent vertex 99.
    src_elems = np.array([
        [0, 1, 2], [0, 2, 3], [0, 1, 99],
    ])
    body_elems = np.array([[4, 5, 0]])
    elements = np.vstack([src_elems, body_elems])
    tags = np.array([2, 2, 2, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)
    # Should not raise, should converge to +z axis from the valid tris.
    frame = infer_frame(grid, tags, source_tag=2, origin_at="mouth")
    assert frame.axis[2] > 0.9


# ---------------------------------------------------------------------------
# Symmetry-plane projection
# ---------------------------------------------------------------------------

def test_yz_symmetry_projects_origin_x_to_zero():
    """Half-mesh in X>=0 (yz symmetry) — origin x-coord must collapse
    to 0 so observation points sit on the symmetry plane.
    """
    # Source at X>0 (half mesh)
    src_verts = np.array([
        [0.05, -0.02, 0.0], [0.15, -0.02, 0.0],
        [0.15, 0.02, 0.0], [0.05, 0.02, 0.0],
    ])
    mouth_verts = np.array([
        [0.05, -0.2, 1.0], [0.25, -0.2, 1.0],
        [0.25, 0.2, 1.0], [0.05, 0.2, 1.0],
    ])
    vertices = np.vstack([src_verts, mouth_verts])
    src_elems = np.array([[0, 1, 2], [0, 2, 3]])
    body_elems = np.array([[4, 5, 6], [4, 6, 7]])
    elements = np.vstack([src_elems, body_elems])
    tags = np.array([2, 2, 1, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)

    frame_default = infer_frame(grid, tags, source_tag=2, origin_at="mouth")
    frame_yz = infer_frame(
        grid, tags, source_tag=2, origin_at="mouth", symmetry_plane="yz",
    )

    # Without symmetry projection, origin.x > 0
    assert frame_default.origin[0] > 0.0
    # With yz symmetry, origin.x = 0
    assert abs(frame_yz.origin[0]) < 1e-12
    # Other coords preserved
    np.testing.assert_allclose(frame_yz.origin[1:], frame_default.origin[1:])


def test_xy_symmetry_projects_origin_z_to_zero():
    """Half-mesh in Z>=0 (xy symmetry)."""
    src_verts = np.array([
        [-0.02, 0.0, 0.05], [0.02, 0.0, 0.05],
        [0.02, 0.0, 0.15], [-0.02, 0.0, 0.15],
    ])
    mouth_verts = np.array([
        [-0.2, 1.0, 0.05], [0.2, 1.0, 0.05],
        [0.2, 1.0, 0.25], [-0.2, 1.0, 0.25],
    ])
    vertices = np.vstack([src_verts, mouth_verts])
    src_elems = np.array([[0, 1, 2], [0, 2, 3]])
    body_elems = np.array([[4, 5, 6], [4, 6, 7]])
    elements = np.vstack([src_elems, body_elems])
    tags = np.array([2, 2, 1, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)
    frame_default = infer_frame(grid, tags, source_tag=2, origin_at="mouth")
    frame_xy = infer_frame(
        grid, tags, source_tag=2, origin_at="mouth", symmetry_plane="xy",
    )

    assert frame_default.origin[2] > 0.0
    assert abs(frame_xy.origin[2]) < 1e-12
    np.testing.assert_allclose(frame_xy.origin[:2], frame_default.origin[:2])


def test_symmetry_plane_none_is_passthrough():
    """``symmetry_plane=None`` must return the same origin as omitting it."""
    src_verts = np.array([
        [0.05, -0.02, 0.0], [0.15, -0.02, 0.0],
        [0.15, 0.02, 0.0], [0.05, 0.02, 0.0],
    ])
    mouth_verts = np.array([
        [0.05, -0.2, 1.0], [0.25, -0.2, 1.0],
        [0.25, 0.2, 1.0], [0.05, 0.2, 1.0],
    ])
    vertices = np.vstack([src_verts, mouth_verts])
    src_elems = np.array([[0, 1, 2], [0, 2, 3]])
    body_elems = np.array([[4, 5, 6], [4, 6, 7]])
    elements = np.vstack([src_elems, body_elems])
    tags = np.array([2, 2, 1, 1], dtype=np.int32)

    grid = _mock_grid(vertices.T, elements.T)
    frame_default = infer_frame(grid, tags, source_tag=2, origin_at="mouth")
    frame_none = infer_frame(
        grid, tags, source_tag=2, origin_at="mouth", symmetry_plane=None,
    )

    np.testing.assert_allclose(frame_none.origin, frame_default.origin)


# ---------------------------------------------------------------------------
# Symmetry-reduced curved source caps
# ---------------------------------------------------------------------------

def test_yz_xz_quadrant_projects_axis_frame_and_origin():
    verts, elems, tags = _build_curved_cap_horn_mesh()
    verts, elems, tags = _restrict_mesh(
        verts, elems, tags,
        lambda centroid: centroid[0] > 0.0 and centroid[1] > 0.0,
    )
    grid = _mock_grid(verts, elems)

    frame = infer_frame(grid, tags, source_tag=2, symmetry_plane="yz+xz")

    np.testing.assert_allclose(frame.axis, [0.0, 0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(frame.u, [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(frame.v, [0.0, 1.0, 0.0], atol=1e-9)
    assert frame.origin[0] == 0.0
    assert frame.origin[1] == 0.0


def test_yz_half_mesh_projects_axis_and_origin():
    verts, elems, tags = _build_curved_cap_horn_mesh()
    verts, elems, tags = _restrict_mesh(
        verts, elems, tags,
        lambda centroid: centroid[0] > 0.0,
    )
    grid = _mock_grid(verts, elems)

    frame = infer_frame(grid, tags, source_tag=2, symmetry_plane="yz")

    np.testing.assert_allclose(frame.axis, [0.0, 0.0, 1.0], atol=1e-9)
    assert frame.origin[0] == 0.0


def test_xz_half_mesh_projects_axis_and_origin():
    verts, elems, tags = _build_curved_cap_horn_mesh()
    verts, elems, tags = _restrict_mesh(
        verts, elems, tags,
        lambda centroid: centroid[1] > 0.0,
    )
    grid = _mock_grid(verts, elems)

    frame = infer_frame(grid, tags, source_tag=2, symmetry_plane="xz")

    np.testing.assert_allclose(frame.axis, [0.0, 0.0, 1.0], atol=1e-9)
    assert frame.origin[1] == 0.0


def test_full_mesh_axis_matches_yz_xz_quadrant_axis():
    full_verts, full_elems, full_tags = _build_curved_cap_horn_mesh()
    full_grid = _mock_grid(full_verts, full_elems)
    full_frame = infer_frame(full_grid, full_tags, source_tag=2)

    quad_verts, quad_elems, quad_tags = _restrict_mesh(
        full_verts, full_elems, full_tags,
        lambda centroid: centroid[0] > 0.0 and centroid[1] > 0.0,
    )
    quad_grid = _mock_grid(quad_verts, quad_elems)
    quad_frame = infer_frame(
        quad_grid, quad_tags, source_tag=2, symmetry_plane="yz+xz",
    )

    np.testing.assert_allclose(full_frame.axis, quad_frame.axis, atol=1e-9)


def test_xy_half_mesh_for_y_firing_horn_projects_axis_to_xy_plane():
    verts, elems, tags = _build_curved_cap_horn_mesh()
    verts = _rotate_z_horn_to_y(verts)
    verts, elems, tags = _restrict_mesh(
        verts, elems, tags,
        lambda centroid: centroid[2] >= 0.0,
    )
    grid = _mock_grid(verts, elems)

    frame = infer_frame(grid, tags, source_tag=2, symmetry_plane="xy")

    assert frame.axis[2] == 0.0
    np.testing.assert_allclose(frame.axis, [0.0, 1.0, 0.0], atol=1e-9)


def test_degenerate_axis_projection_keeps_unit_unprojected_axis():
    verts, elems, tags = _build_curved_cap_horn_mesh()
    grid = _mock_grid(verts, elems)

    frame = infer_frame(grid, tags, source_tag=2, symmetry_plane="xy")

    np.testing.assert_allclose(np.linalg.norm(frame.axis), 1.0, atol=1e-12)
    assert frame.axis[2] > 0.9
