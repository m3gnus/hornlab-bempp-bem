"""Declaring a symmetry plane must not switch off the reduced-mesh detector.

It used to: ``load_mesh`` only ran ``_warn_if_reduced_symmetry_mesh`` when no
plane was declared, which is exactly the case where the detector is least
useful -- it already returns the right answer, it was simply never consulted.
A quarter mesh declared as a half model is mirrored in one axis only, leaving
the other cut plane open, and nothing downstream notices: the expansion's own
edge-count invariant balances because the unmirrored cut reads as a rim.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from hornlab_bempp_bem.mesh import MeshError, _check_declared_symmetry_plane

_R2 = 0.70710678

# A quarter of a closed cone -- apex, base centre, and a quarter of the base
# ring. Closed apart from the x=0 and y=0 cuts, which is what the detector
# requires: every open boundary edge must be explained by a candidate plane.
_QUARTER_VERTICES = np.array(
    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
     [_R2, _R2, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
_QUARTER_TRIANGLES = np.array(
    [[0, 2, 3], [0, 3, 4], [1, 3, 2], [1, 4, 3]], dtype=np.int32,
)

# The same cone mirrored across xz, so it is cut on x=0 only: a genuine 'yz'
# half model.
_HALF_VERTICES = np.array(
    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
     [_R2, _R2, 0.0], [0.0, 1.0, 0.0], [_R2, -_R2, 0.0], [0.0, -1.0, 0.0]],
    dtype=np.float64,
)
_HALF_TRIANGLES = np.array(
    [[0, 2, 3], [0, 3, 4], [1, 3, 2], [1, 4, 3],
     [0, 5, 2], [0, 6, 5], [1, 2, 5], [1, 5, 6]], dtype=np.int32,
)


def _check(vertices, triangles, declared):
    _check_declared_symmetry_plane(vertices, triangles, declared)


@pytest.mark.parametrize("declared", ["xz", "yz"])
def test_quarter_mesh_declared_as_a_half_model_is_rejected(declared):
    with pytest.raises(MeshError, match="would not be mirrored"):
        _check(_QUARTER_VERTICES, _QUARTER_TRIANGLES, declared)


def test_the_error_names_the_unmirrored_plane():
    with pytest.raises(MeshError) as excinfo:
        _check(_QUARTER_VERTICES, _QUARTER_TRIANGLES, "xz")

    assert "'yz'" in str(excinfo.value)


def test_matching_declarations_are_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check(_QUARTER_VERTICES, _QUARTER_TRIANGLES, "yz+xz")
        _check(_HALF_VERTICES, _HALF_TRIANGLES, "yz")


def test_over_declaring_warns_rather_than_raising():
    """A half mesh declared as a quarter is caught by the plane-contact check."""
    with pytest.warns(RuntimeWarning, match="looks cut on"):
        _check(_HALF_VERTICES, _HALF_TRIANGLES, "yz+xz")


def test_undetectable_mesh_is_left_alone():
    """The detector is conservative; no detection means it has no opinion."""
    closed = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    tetra = np.array(
        [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check(closed, tetra, "yz+xz")


def test_the_detector_still_agrees_with_the_fixtures():
    """Guards the fixtures themselves; a silent detector would void this file."""
    from hornlab_bempp_bem.mesh import detect_reduced_symmetry_plane

    assert detect_reduced_symmetry_plane(
        _QUARTER_VERTICES, _QUARTER_TRIANGLES,
    ) == "yz+xz"
    assert detect_reduced_symmetry_plane(
        _HALF_VERTICES, _HALF_TRIANGLES,
    ) == "yz"
