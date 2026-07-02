from __future__ import annotations

import numpy as np
import pytest

from hornlab_bempp_bem.mesh import (
    MeshError,
    _signed_mesh_volume_indicator,
    _validate_outward_normals,
)


def _tetrahedron() -> tuple[np.ndarray, np.ndarray]:
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    outward_tris = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )
    return verts, outward_tris


def test_validate_outward_normals_accepts_canonical_winding():
    verts, tris = _tetrahedron()

    _validate_outward_normals(verts, tris)

    assert _signed_mesh_volume_indicator(verts, tris) > 0


def test_validate_outward_normals_rejects_inward_winding_by_default():
    verts, outward = _tetrahedron()
    inward = outward[:, [0, 2, 1]].copy()

    with pytest.raises(MeshError, match="Canonical meshes"):
        _validate_outward_normals(verts, inward)

    assert _signed_mesh_volume_indicator(verts, inward) < 0


def test_validate_outward_normals_repairs_only_when_explicit():
    verts, outward = _tetrahedron()
    inward = outward[:, [0, 2, 1]].copy()

    _validate_outward_normals(verts, inward, repair=True)

    assert _signed_mesh_volume_indicator(verts, inward) > 0


def _half_cube() -> tuple[np.ndarray, np.ndarray]:
    # Cube surface cut at x=0 keeping x >= 0: open rim is exactly the x=0
    # square — the canonical mirror-reduced (half) mesh shape.
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    tris = np.array(
        [
            [0, 3, 2],
            [0, 2, 1],  # z=0, outward -z
            [4, 5, 6],
            [4, 6, 7],  # z=1, outward +z
            [0, 1, 5],
            [0, 5, 4],  # y=0, outward -y
            [3, 7, 6],
            [3, 6, 2],  # y=1, outward +y
            [1, 2, 6],
            [1, 6, 5],  # x=1, outward +x
        ],
        dtype=np.int32,
    )
    return verts, tris


def test_open_boundary_edges_closed_vs_open():
    from hornlab_bempp_bem.mesh import open_boundary_edges

    verts, tris = _tetrahedron()
    assert open_boundary_edges(tris).size == 0
    assert open_boundary_edges(tris[:-1]).shape == (3, 2)


def test_detect_reduced_symmetry_plane_flags_half_cube():
    from hornlab_bempp_bem.mesh import detect_reduced_symmetry_plane

    verts, tris = _half_cube()
    assert detect_reduced_symmetry_plane(verts, tris) == "yz"


def test_detect_reduced_symmetry_plane_ignores_closed_mesh():
    from hornlab_bempp_bem.mesh import detect_reduced_symmetry_plane

    verts, tris = _tetrahedron()
    assert detect_reduced_symmetry_plane(verts, tris) is None


def test_reduced_mesh_warning_points_to_metal_backend():
    from hornlab_bempp_bem.mesh import _warn_if_reduced_symmetry_mesh

    verts, tris = _half_cube()
    with pytest.warns(RuntimeWarning, match="native_symmetry_plane"):
        _warn_if_reduced_symmetry_mesh(verts, tris)


def _tet_msh_text(*, drop_wall_face: bool) -> str:
    # 1-based faces of the outward tetrahedron; face 4 carries source tag 2.
    faces = [
        ("1 3 2", 1),
        ("1 2 4", 1),
        ("1 4 3", 1),
        ("2 3 4", 2),
    ]
    if drop_wall_face:
        faces = faces[:1] + faces[2:]
    lines = [
        "$MeshFormat",
        "2.2 0 8",
        "$EndMeshFormat",
        "$Nodes",
        "4",
        "1 0 0 0",
        "2 1 0 0",
        "3 0 1 0",
        "4 0 0 1",
        "$EndNodes",
        "$Elements",
        str(len(faces)),
    ]
    for index, (nodes, phys) in enumerate(faces, start=1):
        lines.append(f"{index} 2 2 {phys} 1 {nodes}")
    lines += ["$EndElements", ""]
    return "\n".join(lines)


def test_load_mesh_require_closed(tmp_path):
    from hornlab_bempp_bem.mesh import load_mesh

    closed = tmp_path / "tet.msh"
    closed.write_text(_tet_msh_text(drop_wall_face=False))
    load_mesh(closed, require_closed=True)

    leaking = tmp_path / "tet-open.msh"
    leaking.write_text(_tet_msh_text(drop_wall_face=True))
    with pytest.raises(MeshError, match="open boundary edges"):
        load_mesh(leaking, require_closed=True)
