"""A reduced mesh whose cut boundary misses the mirror plane must not solve.

The symmetry reduction itself is exact, so the dangerous failures are all
geometric: a mesh that is *nearly* on its mirror plane tears open along the
seam when mirrored, and a mesh carrying a rigid offset mirrors into two
disjoint shells. Both produce a plausible-looking answer that is wrong by
around 10% in impedance, and neither is caught anywhere else unless the caller
happens to have set ``require_closed_mesh``.
"""
from __future__ import annotations

import numpy as np
import pytest

from hornlab_bempp_bem.mesh import open_boundary_edges
from hornlab_bempp_bem.symmetry import (
    _require_seamless_expansion,
    expand_symmetry_mesh,
)

# An open square pyramid sitting on the origin: four side faces, no base, so it
# has a genuine open rim as well as two cut boundaries. Quarter-reduced, it
# occupies x >= 0, y >= 0.
_APEX = [0.0, 0.0, 1.0]
_QUARTER_VERTICES = np.array(
    [_APEX, [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_QUARTER_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
_QUARTER_TAGS = np.array([2, 2], dtype=np.int32)


def _expand(vertices, plane="yz+xz", **kwargs):
    return expand_symmetry_mesh(
        vertices, _QUARTER_TRIANGLES, _QUARTER_TAGS, plane, **kwargs
    )


def test_healthy_quarter_mesh_expands_without_new_open_edges():
    expanded = _expand(_QUARTER_VERTICES)

    assert expanded.triangles_nx3.shape == (8, 3)
    # Only the base rim stays open: 8 edges, one per mirrored side face.
    open_edges = open_boundary_edges(expanded.triangles_nx3.astype(np.int32))
    assert open_edges.shape[0] == 8


def test_seam_within_the_plane_tolerance_is_snapped_and_welds_exactly():
    """A seam off by less than the tolerance must weld, not tear."""
    displaced = _QUARTER_VERTICES.copy()
    displaced[displaced[:, 0] == 0.0, 0] = 1.0e-9
    displaced[displaced[:, 1] == 0.0, 1] = 1.0e-9

    expanded = _expand(displaced)
    reference = _expand(_QUARTER_VERTICES)

    assert np.array_equal(expanded.triangles_nx3, reference.triangles_nx3)
    assert expanded.vertices_nx3.shape == reference.vertices_nx3.shape
    assert open_boundary_edges(
        expanded.triangles_nx3.astype(np.int32)
    ).shape[0] == 8


def test_seam_beyond_the_plane_tolerance_is_rejected_not_torn():
    """Past the tolerance the whole seam lifts off, so the mesh is refused."""
    displaced = _QUARTER_VERTICES.copy()
    displaced[displaced[:, 0] == 0.0, 0] = 1.0e-3
    displaced[displaced[:, 1] == 0.0, 1] = 1.0e-3

    with pytest.raises(ValueError, match="does not touch its mirror plane"):
        _expand(displaced, plane_tolerance=1.0e-6)


def test_unwelded_expansion_is_rejected_by_the_edge_count_invariant():
    """Backstop for a seam that survives snapping without merging.

    Plane snapping and the touch check between them make this unreachable from
    geometry alone, so exercise the invariant directly: four unmerged image
    blocks leave 16 open edges where a welded quarter model has 8.
    """
    unwelded = np.vstack([
        _QUARTER_TRIANGLES + image * _QUARTER_VERTICES.shape[0]
        for image in range(4)
    ])

    with pytest.raises(ValueError, match="open boundary edges"):
        _require_seamless_expansion(
            _QUARTER_VERTICES,
            _QUARTER_TRIANGLES,
            unwelded,
            "yz+xz",
            image_count=4,
            plane_tolerance=1.0e-6,
        )


def test_the_invariants_are_actually_wired_into_expansion(monkeypatch):
    """Both backstops must run, and both must respect validate_seam.

    They are unreachable from ordinary geometry once snapping and the
    cut-boundary check are in place, so nothing else would notice if the calls
    were dropped.
    """
    import hornlab_bempp_bem.symmetry as symmetry

    called: list[str] = []
    for name in ("_require_seamless_expansion", "_require_cut_boundary"):
        original = getattr(symmetry, name)

        def wrapper(*args, _name=name, _original=original, **kwargs):
            called.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(symmetry, name, wrapper)

    _expand(_QUARTER_VERTICES)
    assert sorted(called) == [
        "_require_cut_boundary", "_require_seamless_expansion",
    ]

    called.clear()
    _expand(_QUARTER_VERTICES, validate_seam=False)
    assert called == []


def test_edge_count_invariant_accepts_a_correctly_welded_expansion():
    expanded = _expand(_QUARTER_VERTICES)

    _require_seamless_expansion(
        _QUARTER_VERTICES,
        _QUARTER_TRIANGLES,
        expanded.triangles_nx3,
        "yz+xz",
        image_count=4,
        plane_tolerance=1.0e-6,
    )


def test_caller_can_loosen_the_seam_tolerance():
    """The tolerance used to be clamped to 1e-9, which callers could not raise."""
    displaced = _QUARTER_VERTICES.copy()
    displaced[displaced[:, 0] == 0.0, 0] = 1.0e-4
    displaced[displaced[:, 1] == 0.0, 1] = 1.0e-4

    expanded = _expand(displaced, plane_tolerance=1.0e-3)

    assert np.array_equal(
        expanded.triangles_nx3, _expand(_QUARTER_VERTICES).triangles_nx3,
    )


def test_mesh_offset_away_from_the_mirror_plane_is_rejected():
    """The mesher applies VerticalOffset after cut-plane snapping."""
    offset = _QUARTER_VERTICES.copy()
    offset[:, 1] += 0.05

    with pytest.raises(ValueError, match="does not touch its mirror plane"):
        _expand(offset)


def test_mesh_on_the_wrong_side_of_the_plane_is_still_rejected():
    flipped = _QUARTER_VERTICES.copy()
    flipped[:, 0] -= 1.0

    with pytest.raises(ValueError, match="must lie in X >= 0"):
        _expand(flipped)


def test_half_model_offset_in_the_unmirrored_axis_is_allowed():
    """Only the mirrored axis is constrained; 'yz' says nothing about y."""
    shifted = _QUARTER_VERTICES.copy()
    shifted[:, 1] += 0.05

    expanded = _expand(shifted, plane="yz")

    assert expanded.triangles_nx3.shape == (4, 3)


def test_one_lifted_seam_vertex_is_rejected():
    """The mesh still touches the plane, so only the cut boundary sees this.

    Lifting a single seam vertex tears the surface along the cut, and the
    edge-count invariant cannot notice because the affected edge simply
    reclassifies as an ordinary rim.
    """
    torn = _QUARTER_VERTICES.copy()
    on_x_plane = np.flatnonzero(torn[:, 0] == 0.0)
    torn[on_x_plane[-1], 0] = 1.0e-3

    with pytest.raises(ValueError, match="no open boundary edge lying in it"):
        _expand(torn, plane_tolerance=1.0e-6)


def test_a_body_merely_tangent_to_the_plane_is_rejected():
    """A closed shell touching at one vertex mirrors into two joined pieces.

    ``min(coord) == 0`` is satisfied and neither shell has an open edge, so
    both the touch check and the edge count accept it.
    """
    tetra_vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.0, 1.5, 0.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    tetra = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64,
    )

    with pytest.raises(ValueError, match="no open boundary edge lying in it"):
        expand_symmetry_mesh(
            tetra_vertices, tetra, np.array([2, 2, 2, 2], dtype=np.int32), "yz",
        )


def test_validate_seam_can_be_disabled_for_diagnostics():
    """The diagnostics script needs to expand a suspect mesh to inspect it."""
    torn = _QUARTER_VERTICES.copy()
    on_x_plane = np.flatnonzero(torn[:, 0] == 0.0)
    torn[on_x_plane[-1], 0] = 1.0e-3

    expanded = _expand(torn, plane_tolerance=1.0e-6, validate_seam=False)

    assert expanded.triangles_nx3.shape == (8, 3)


def test_snapping_that_would_collapse_a_triangle_is_rejected():
    """Snapping moves a vertex by at most the tolerance; that must stay safe.

    It is only unsafe when two vertices are already closer together than the
    tolerance, which is a broken mesh rather than a near-plane seam.
    """
    degenerate = np.vstack([
        _QUARTER_VERTICES,
        # Two vertices 1e-9 apart in x at the same y and z: both snap to
        # exactly zero, becoming coincident, and the triangle loses its area.
        np.array([[1.0e-9, 0.5, 0.5], [2.0e-9, 0.5, 0.5]]),
    ])
    triangles = np.vstack([
        _QUARTER_TRIANGLES, np.array([[1, 4, 5]], dtype=np.int64),
    ])
    tags = np.array([2, 2, 2], dtype=np.int32)

    with pytest.raises(ValueError, match="collapsed 1 triangle"):
        expand_symmetry_mesh(degenerate, triangles, tags, "yz+xz")


def test_a_healthy_mesh_is_never_collapsed_by_snapping():
    displaced = _QUARTER_VERTICES.copy()
    displaced[displaced[:, 0] == 0.0, 0] = 1.0e-9

    expanded = _expand(displaced)

    assert expanded.triangles_nx3.shape == (8, 3)


def test_expansion_does_not_mutate_the_caller_s_vertices():
    """Plane snapping must not be visible to the caller."""
    displaced = _QUARTER_VERTICES.copy()
    displaced[displaced[:, 0] == 0.0, 0] = 1.0e-9
    before = displaced.copy()

    _expand(displaced)

    assert np.array_equal(displaced, before)
