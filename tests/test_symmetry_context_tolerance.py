"""The seam tolerance must work through ``build_symmetry_context``, not just
through ``expand_symmetry_mesh``.

Expansion snaps near-plane coordinates to exactly zero, but the orbit map is
built by matching reduced P1 dof coordinates against the expanded grid. If the
reduced side is not snapped identically the lookup misses by up to the
tolerance and the context raises ``missing full dofs`` -- so a seam the
expansion happily repaired becomes a hard failure one call later. These pin the
whole advertised band, not just the expansion step.
"""
from __future__ import annotations

import numpy as np
import pytest

bempp_api = pytest.importorskip("bempp_cl.api")

from hornlab_bempp_bem.symmetry import build_symmetry_context

# Quarter of a closed cone: apex, base centre, quarter base ring. Closed apart
# from the x=0 and y=0 cuts, so both mirror planes carry a real cut boundary.
_R2 = 0.70710678
_VERTICES = np.array(
    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
     [_R2, _R2, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_TRIANGLES = np.array(
    [[0, 2, 3], [0, 3, 4], [1, 3, 2], [1, 4, 3]], dtype=np.uint32,
)
_TAGS = np.array([1, 1, 2, 2], dtype=np.int32)


def _context(vertices, **kwargs):
    grid = bempp_api.Grid(
        np.ascontiguousarray(vertices.T),
        np.ascontiguousarray(_TRIANGLES.T),
    )
    return build_symmetry_context(grid, _TAGS, "yz+xz", **kwargs)


def _displaced(amount):
    vertices = _VERTICES.copy()
    vertices[np.abs(vertices[:, 0]) < 1e-15, 0] = amount
    vertices[np.abs(vertices[:, 1]) < 1e-15, 1] = amount
    return vertices


def test_a_healthy_quarter_builds_a_context():
    context = _context(_VERTICES)

    assert context.image_count == 4
    assert context.active_dof_count > 0
    assert context.pressure_expansion.shape[1] == context.active_dof_count


@pytest.mark.parametrize("displacement", [1e-12, 1e-9, 1e-8, 1e-7, 5e-7, 1e-6])
def test_the_whole_advertised_tolerance_band_works_end_to_end(displacement):
    """Anything inside plane_tolerance must build, not just the tiniest cases."""
    reference = _context(_VERTICES)

    context = _context(_displaced(displacement))

    assert context.active_dof_count == reference.active_dof_count
    assert context.image_count == reference.image_count
    np.testing.assert_array_equal(
        context.pressure_expansion.toarray(),
        reference.pressure_expansion.toarray(),
    )


def test_a_displacement_past_the_tolerance_is_refused_not_mismatched():
    """It must fail as a geometry error, not as an orbit bookkeeping error."""
    with pytest.raises(ValueError) as excinfo:
        _context(_displaced(1e-3))

    assert "missing full dofs" not in str(excinfo.value)


def test_a_caller_supplied_tolerance_also_reaches_the_orbit_map():
    reference = _context(_VERTICES)

    context = _context(_displaced(1e-4), plane_tolerance=1e-3)

    assert context.active_dof_count == reference.active_dof_count
