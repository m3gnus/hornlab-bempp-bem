from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_bempp_bem import _resolve_mesh
from hornlab_bempp_bem.mesh import (
    LoadedMesh,
    MeshError,
    _signed_mesh_volume_indicator,
    _validate_outward_normals,
)
from hornlab_bempp_bem.result import MeshInfo


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


def _tet_msh_text(*, drop_wall_face: bool, duplicate_wall_face: bool = False) -> str:
    # 1-based faces of the outward tetrahedron; face 4 carries source tag 2.
    faces = [
        ("1 3 2", 1),
        ("1 2 4", 1),
        ("1 4 3", 1),
        ("2 3 4", 2),
    ]
    if drop_wall_face:
        faces = faces[:1] + faces[2:]
    if duplicate_wall_face:
        faces.append(faces[0])
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
    from hornlab_bempp_bem.mesh import _require_closed_surface, load_mesh

    closed = tmp_path / "tet.msh"
    closed.write_text(_tet_msh_text(drop_wall_face=False))
    load_mesh(closed, require_closed=True)

    leaking = tmp_path / "tet-open.msh"
    leaking.write_text(_tet_msh_text(drop_wall_face=True))
    with pytest.raises(MeshError, match="open boundary edges"):
        load_mesh(leaking, require_closed=True)
    with pytest.raises(MeshError, match="open boundary edges"):
        load_mesh(leaking, validate=False, require_closed=True)

    verts, tris = _tetrahedron()
    duplicated = np.vstack([tris, tris[0:1]])
    with pytest.raises(MeshError, match="non-manifold"):
        _require_closed_surface(verts, duplicated)

    nonmanifold = tmp_path / "tet-duplicate-face.msh"
    nonmanifold.write_text(_tet_msh_text(drop_wall_face=False, duplicate_wall_face=True))
    with pytest.raises(MeshError, match="non-manifold"):
        load_mesh(nonmanifold, require_closed=True)


def test_resolve_loaded_mesh_require_closed_rechecks_boundaries():
    verts, tris = _tetrahedron()
    open_tris = tris[:-1].copy()
    loaded = LoadedMesh(
        grid=SimpleNamespace(vertices=verts.T, elements=open_tris.T),
        physical_tags=np.ones(open_tris.shape[0], dtype=np.int32),
        info=MeshInfo(
            n_vertices=len(verts),
            n_triangles=len(open_tris),
            physical_groups={1: "wall"},
            bounding_box_m=(verts.min(axis=0), verts.max(axis=0)),
        ),
    )

    assert _resolve_mesh(loaded, require_closed=False) is loaded
    with pytest.raises(MeshError, match="open boundary edges"):
        _resolve_mesh(loaded, require_closed=True)

    duplicated = np.vstack([tris, tris[0:1]])
    loaded_nonmanifold = LoadedMesh(
        grid=SimpleNamespace(vertices=verts.T, elements=duplicated.T),
        physical_tags=np.ones(duplicated.shape[0], dtype=np.int32),
        info=MeshInfo(
            n_vertices=len(verts),
            n_triangles=len(duplicated),
            physical_groups={1: "wall"},
            bounding_box_m=(verts.min(axis=0), verts.max(axis=0)),
        ),
    )
    with pytest.raises(MeshError, match="non-manifold"):
        _resolve_mesh(loaded_nonmanifold, require_closed=True)
