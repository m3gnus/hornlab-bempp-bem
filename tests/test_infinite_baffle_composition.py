"""Unsupported compositions are refused on otherwise valid IB geometry."""

from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_bempp_bem.config import SolveConfig, reject_unsupported_native_symmetry
from hornlab_bempp_bem.infinite_baffle import _validate_coupled_infinite_baffle


def _cavity():
    vertices = np.array(
        [
            [1, 1, -1],
            [2, 1, -1],
            [2, 2, -1],
            [1, 2, -1],
            [1, 1, 0],
            [2, 1, 0],
            [2, 2, 0],
            [1, 2, 0],
        ],
        dtype=float,
    )
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )[:, [0, 2, 1]]
    return SimpleNamespace(
        grid=SimpleNamespace(vertices=vertices.T, elements=triangles.T),
        physical_tags=np.array([2, 2, 12, 12] + [1] * 8),
    )


@pytest.mark.parametrize("ground_plane", ["yz", "xz", "xy"])
def test_valid_coupled_ib_cavity_refuses_ground_plane(ground_plane):
    mesh = _cavity()
    frame = SimpleNamespace(axis=np.array([0.0, 0.0, 1.0]))
    geometry = _validate_coupled_infinite_baffle(
        mesh,
        SolveConfig(aperture_tag=12),
        frame,
    )
    assert geometry.element_indices.tolist() == [2, 3]

    config = SolveConfig(aperture_tag=12, ground_plane=ground_plane)
    reject_unsupported_native_symmetry(config)
    with pytest.raises(NotImplementedError, match="ground_plane.*Remove ground_plane"):
        _validate_coupled_infinite_baffle(mesh, config, frame)
