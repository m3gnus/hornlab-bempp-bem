from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from hornlab_bempp_bem import _resolve_mesh
from hornlab_bempp_bem.mesh import (
    LoadedMesh,
    MeshError,
    _merge_duplicate_vertices,
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


def test_open_surface_winding_verdict_is_translation_invariant():
    verts = np.array(
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        dtype=np.float64,
    )
    tris = np.array([[0, 1, 2]], dtype=np.int32)
    translated = verts + np.array([0.0, 0.0, -2.0])
    assert _signed_mesh_volume_indicator(verts, tris) > 0.0
    assert _signed_mesh_volume_indicator(translated, tris) < 0.0

    original = tris.copy()
    _validate_outward_normals(verts, tris, repair=True)
    _validate_outward_normals(translated, tris, repair=True)
    np.testing.assert_array_equal(tris, original)


def test_duplicate_merge_uses_actual_euclidean_distance():
    triangles = np.array([[0, 1, 2]], dtype=np.int32)
    farther_than_tol = np.array(
        [[0.49, 0.49, 0.49], [-0.49, -0.49, -0.49], [5.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    merged_verts, merged_tris, count = _merge_duplicate_vertices(
        farther_than_tol, triangles, 1.0
    )
    assert count == 0
    assert len(merged_verts) == 3
    np.testing.assert_array_equal(merged_tris, triangles)

    closer_than_tol = np.array(
        [[0.49, 0.0, 0.0], [0.51, 0.0, 0.0], [5.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    merged_verts, merged_tris, count = _merge_duplicate_vertices(
        closer_than_tol, triangles, 1.0
    )
    assert count == 1
    assert len(merged_verts) == 2
    assert merged_tris[0, 0] == merged_tris[0, 1]


def _reference_merge_duplicate_vertices(
    verts: np.ndarray,
    tris: np.ndarray,
    tol: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Pre-cKDTree spatial-hash merger retained as an equivalence oracle."""
    cells = np.floor(verts / tol).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    for index, key in enumerate(map(tuple, cells)):
        buckets.setdefault(key, []).append(index)

    parent = np.arange(len(verts), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]
    tol_sq = float(tol) ** 2
    for key, indices in buckets.items():
        neighbours = [
            neighbour
            for dx, dy, dz in offsets
            for neighbour in buckets.get((key[0] + dx, key[1] + dy, key[2] + dz), ())
        ]
        for left in indices:
            for right in neighbours:
                if right <= left:
                    continue
                delta = verts[right] - verts[left]
                if float(delta @ delta) > tol_sq:
                    continue
                root_left = find(left)
                root_right = find(right)
                if root_left != root_right:
                    parent[max(root_left, root_right)] = min(root_left, root_right)

    roots = np.fromiter(
        (find(index) for index in range(len(verts))),
        dtype=np.int64,
        count=len(verts),
    )
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    if len(unique_roots) == len(verts):
        return verts, tris, 0
    return (
        verts[unique_roots],
        inverse[tris].astype(np.int32, copy=False),
        len(verts) - len(unique_roots),
    )


def test_duplicate_merge_matches_spatial_hash_reference_on_edge_fixtures():
    vertices = np.array(
        [
            [30.9, 0.0, 0.0],
            [10.49, 10.49, 10.49],
            [0.0, 0.0, 0.0],
            [40.0, 0.0, 0.0],
            [21.0, 0.0, 0.0],
            [31.8, 0.0, 0.0],
            [-0.6369326152038236, -0.7154417306587971, -0.28715844706635957],
            [40.0, 0.0, 0.0],
            [20.49, 0.0, 0.0],
            [9.51, 9.51, 9.51],
            [30.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [[2, 0, 4], [8, 7, 5], [1, 9, 3], [10, 6, 4]],
        dtype=np.int32,
    )

    expected = _reference_merge_duplicate_vertices(vertices, triangles, 1.0)
    actual = _merge_duplicate_vertices(vertices, triangles, 1.0)

    assert actual[2] == expected[2]
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])


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


def test_load_mesh_reuses_edge_incidence_for_validation(monkeypatch, tmp_path):
    import hornlab_bempp_bem.mesh as mesh_module

    mesh_path = tmp_path / "tet.msh"
    mesh_path.write_text(_tet_msh_text(drop_wall_face=False))
    original = mesh_module._edge_incidence_counts
    call_count = 0

    def count_calls(triangles):
        nonlocal call_count
        call_count += 1
        return original(triangles)

    monkeypatch.setattr(mesh_module, "_edge_incidence_counts", count_calls)

    mesh_module.load_mesh(mesh_path, require_closed=True, validate=True)

    assert call_count == 1


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
