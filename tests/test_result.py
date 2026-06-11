from __future__ import annotations

import numpy as np

from hornlab_bempp_bem.config import SolveConfig
from hornlab_bempp_bem.result import MeshInfo, SolveResult


def test_directivity_db_alias_returns_spl_db():
    spl_db = np.array([[[0.0, -3.0]]], dtype=np.float64)
    result = SolveResult(
        frequencies_hz=np.array([1000.0], dtype=np.float64),
        pressure_complex=np.ones((1, 1, 2), dtype=np.complex128),
        spl_db=spl_db,
        impedance=np.array([1.0 + 2.0j], dtype=np.complex128),
        observation_angles_deg=np.array([0.0, 30.0], dtype=np.float64),
        observation_points=np.zeros((1, 2, 3), dtype=np.float64),
        observation_planes=["horizontal"],
        config=SolveConfig(),
        mesh_info=MeshInfo(
            n_vertices=0,
            n_triangles=0,
            physical_groups={},
            bounding_box_m=(
                np.zeros(3, dtype=np.float64),
                np.zeros(3, dtype=np.float64),
            ),
        ),
    )

    assert result.directivity_db is spl_db
