from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from hornlab_bempp_bem.symmetry import (
    expand_symmetry_mesh,
    reduce_test_vector,
    reduce_trial_matrix,
)


_VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=np.float64,
)
_TRIANGLES = np.array([[0, 1, 2]], dtype=np.int64)
_TAGS = np.array([2], dtype=np.int32)


def test_quarter_expansion_shares_seams_and_reverses_reflected_winding():
    expanded = expand_symmetry_mesh(
        _VERTICES, _TRIANGLES, _TAGS, "yz+xz",
    )

    assert expanded.image_signs == (
        (1, 1, 1),
        (1, -1, 1),
        (-1, 1, 1),
        (-1, -1, 1),
    )
    assert expanded.triangles_nx3.shape == (4, 3)
    assert expanded.physical_tags.tolist() == [2, 2, 2, 2]
    assert expanded.vertices_nx3.shape[0] < 4 * _VERTICES.shape[0]

    normals = []
    for triangle in expanded.triangles_nx3:
        p0, p1, p2 = expanded.vertices_nx3[triangle]
        normals.append(np.cross(p1 - p0, p2 - p0))
    normals = np.asarray(normals)
    # Reflection plus winding reversal preserves the corresponding normal.
    np.testing.assert_allclose(normals[1], normals[0] * [1, -1, 1])
    np.testing.assert_allclose(normals[2], normals[0] * [-1, 1, 1])
    np.testing.assert_allclose(normals[3], normals[0] * [-1, -1, 1])


@pytest.mark.parametrize(
    ("plane", "image_count"),
    [("yz", 2), ("xz", 2), ("yz+xz", 4)],
)
def test_supported_symmetry_modes_expand_expected_image_count(
    plane,
    image_count,
):
    expanded = expand_symmetry_mesh(
        _VERTICES, _TRIANGLES, _TAGS, plane,
    )
    assert len(expanded.image_signs) == image_count
    assert expanded.triangles_nx3.shape[0] == image_count


def test_expansion_rejects_mesh_on_wrong_side_of_plane():
    vertices = _VERTICES.copy()
    vertices[1, 0] = -1.0
    with pytest.raises(ValueError, match="X >= 0"):
        expand_symmetry_mesh(vertices, _TRIANGLES, _TAGS, "yz")


def test_orbit_reduction_selects_real_rows_and_sums_image_columns():
    context = SimpleNamespace(
        active_test_rows=np.array([0, 2], dtype=np.int64),
        pressure_expansion=sparse.csr_matrix(
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                ]
            )
        ),
    )
    full = np.arange(12, dtype=np.float64).reshape(3, 4)

    reduced = reduce_trial_matrix(full, context)

    np.testing.assert_array_equal(
        reduced,
        [
            [full[0, 0] + full[0, 2], full[0, 1] + full[0, 3]],
            [full[2, 0] + full[2, 2], full[2, 1] + full[2, 3]],
        ],
    )
    np.testing.assert_array_equal(
        reduce_test_vector(np.array([4.0, 5.0, 6.0]), context),
        [4.0, 6.0],
    )
