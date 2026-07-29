from __future__ import annotations

import warnings
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
    _warn_if_inverted_open_shell,
    open_shell_bore_alignment,
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


def _quarter_cube() -> tuple[np.ndarray, np.ndarray]:
    # Unit-cube surface with the x=0 and y=0 faces omitted. Its open edges
    # therefore require both the yz and xz symmetry planes.
    verts, _ = _half_cube()
    tris = np.array(
        [
            [0, 3, 2],
            [0, 2, 1],  # z=0
            [4, 5, 6],
            [4, 6, 7],  # z=1
            [3, 7, 6],
            [3, 6, 2],  # y=1
            [1, 2, 6],
            [1, 6, 5],  # x=1
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


def test_detect_reduced_symmetry_plane_flags_quarter_cube():
    from hornlab_bempp_bem.mesh import detect_reduced_symmetry_plane

    verts, tris = _quarter_cube()
    assert detect_reduced_symmetry_plane(verts, tris) == "yz+xz"


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
        "$PhysicalNames",
        "2",
        '2 1 "rigid wall"',
        '2 2 "velocity source"',
        "$EndPhysicalNames",
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
    loaded = load_mesh(closed, require_closed=True)
    assert loaded.info.physical_groups == {1: "rigid wall", 2: "velocity source"}

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


@pytest.mark.parametrize(
    ("file_format", "binary"),
    [
        ("gmsh22", False),
        ("gmsh22", True),
        ("gmsh", False),
        ("gmsh", True),
    ],
)
def test_load_mesh_preserves_2d_name_reused_by_3d_group(
    tmp_path, file_format, binary
):
    import meshio

    from hornlab_bempp_bem.mesh import load_mesh

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    mesh = meshio.Mesh(
        points,
        [
            ("triangle", np.array([[0, 2, 1]], dtype=np.int32)),
            ("triangle", np.array([[3, 4, 5]], dtype=np.int32)),
        ],
        cell_data={
            "gmsh:physical": [
                np.array([1], dtype=np.int32),
                np.array([2], dtype=np.int32),
            ],
            "gmsh:geometrical": [
                np.array([1], dtype=np.int32),
                np.array([2], dtype=np.int32),
            ],
        },
        point_data={
            "gmsh:dim_tags": np.array(
                [[2, 1]] * 3 + [[2, 2]] * 3,
                dtype=np.int32,
            ),
        },
        field_data={
            "wall": np.array([1, 2], dtype=np.int32),
            "source": np.array([2, 2], dtype=np.int32),
        },
    )
    mesh_path = tmp_path / f"collision-{file_format}-{binary}.msh"
    meshio.write(mesh_path, mesh, file_format=file_format, binary=binary)

    original_names = (
        b'$PhysicalNames\n2\n2 1 "wall"\n2 2 "source"\n$EndPhysicalNames'
    )
    colliding_names = (
        b'$PhysicalNames\n3\n2 1 "wall"\n3 7 "wall"\n'
        b'2 2 "source"\n$EndPhysicalNames'
    )
    contents = mesh_path.read_bytes()
    assert original_names in contents
    mesh_path.write_bytes(contents.replace(original_names, colliding_names, 1))

    loaded = load_mesh(mesh_path, validate=False, merge_tol=0)

    assert loaded.info.physical_groups == {1: "wall", 2: "source"}
    np.testing.assert_array_equal(loaded.physical_tags, [1, 2])


def test_load_mesh_uses_all_triangle_blocks_with_aligned_tags(
    monkeypatch, tmp_path
):
    import meshio

    from hornlab_bempp_bem.mesh import load_mesh

    verts, tris = _tetrahedron()
    mesh = meshio.Mesh(
        verts,
        [
            ("triangle", tris[:2]),
            ("line", np.array([[0, 1]], dtype=np.int32)),
            ("triangle", tris[2:]),
        ],
        cell_data={
            "gmsh:physical": [
                np.array([1, 1], dtype=np.int32),
                np.array([9], dtype=np.int32),
                np.array([2, 3], dtype=np.int32),
            ],
        },
    )
    monkeypatch.setattr(meshio, "read", lambda _path: mesh)
    mesh_path = tmp_path / "multi-block.msh"
    mesh_path.touch()

    loaded = load_mesh(mesh_path, validate=False, merge_tol=0)

    np.testing.assert_array_equal(loaded.grid.elements.T, tris)
    np.testing.assert_array_equal(loaded.physical_tags, [1, 1, 2, 3])


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


def _cone_horn(
    n_phi: int = 24,
    n_axial: int = 6,
    r_throat: float = 0.0127,
    r_mouth: float = 0.15,
    length: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed conical horn: wall (tag 1), throat cap (tag 2), mouth cap (tag 3).

    Wound outward, so the solid's exterior is the positive side. The bare
    variants below are derived from it rather than wound by hand, which keeps
    "into the bore" a consequence of the construction instead of an assertion.
    """
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    radii = r_throat + (r_mouth - r_throat) * np.arange(n_axial + 1) / n_axial
    z = length * np.arange(n_axial + 1) / n_axial
    verts = np.array(
        [
            [radius * np.cos(angle), radius * np.sin(angle), height]
            for radius, height in zip(radii, z, strict=True)
            for angle in phi
        ],
        dtype=np.float64,
    )
    throat_centre = len(verts)
    mouth_centre = throat_centre + 1
    verts = np.vstack([verts, [0.0, 0.0, 0.0], [0.0, 0.0, length]])

    def node(ring: int, k: int) -> int:
        return ring * n_phi + (k % n_phi)

    tris: list[list[int]] = []
    tags: list[int] = []
    for ring in range(n_axial):
        for k in range(n_phi):
            a, b = node(ring, k), node(ring, k + 1)
            c, d = node(ring + 1, k + 1), node(ring + 1, k)
            tris += [[a, b, c], [a, c, d]]
            tags += [1, 1]
    for k in range(n_phi):
        tris.append([throat_centre, node(0, k + 1), node(0, k)])
        tags.append(2)
        tris.append([mouth_centre, node(n_axial, k), node(n_axial, k + 1)])
        tags.append(3)
    return (
        verts,
        np.asarray(tris, dtype=np.int32),
        np.asarray(tags, dtype=np.int32),
    )


def _bare_cone_horn(**kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop the mouth cap and reverse: wall and throat cap now face the bore."""
    verts, tris, tags = _cone_horn(**kwargs)
    keep = tags != 3
    return verts, tris[keep][:, [0, 2, 1]].copy(), tags[keep].copy()


def test_open_shell_bore_alignment_separates_the_two_windings():
    verts, tris, tags = _bare_cone_horn()

    assert open_shell_bore_alignment(verts, tris, tags) == 1.0

    wall = tags == 1
    inverted = tris.copy()
    inverted[wall] = inverted[wall][:, [0, 2, 1]]
    assert open_shell_bore_alignment(verts, inverted, tags) == 0.0


def test_open_shell_bore_alignment_survives_rigid_motion():
    """The measure takes both references from the mesh, so it must be intrinsic."""
    verts, tris, tags = _bare_cone_horn()
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    moved = verts @ rotation.T + np.array([0.4, -1.3, 2.2])

    assert open_shell_bore_alignment(moved, tris, tags) == 1.0


def test_inverted_open_shell_warns():
    verts, tris, tags = _bare_cone_horn()
    tris[tags == 1] = tris[tags == 1][:, [0, 2, 1]]

    with pytest.warns(RuntimeWarning, match="acoustic domain"):
        _warn_if_inverted_open_shell(
            verts, tris, tags
        )


def test_correct_open_shell_does_not_warn():
    verts, tris, tags = _bare_cone_horn()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_inverted_open_shell(
            verts, tris, tags
        )


def test_closed_horn_is_not_judged_as_an_open_shell():
    """A closed body's outer wall legitimately faces away from the bore."""
    verts, tris, tags = _cone_horn()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_inverted_open_shell(
            verts, tris, tags
        )


def test_mirror_reduced_closed_horn_is_not_judged_as_an_open_shell():
    """Reduced bodies are open only on their cut plane; skip, do not warn."""
    verts, tris, tags = _cone_horn()
    keep = np.all(verts[tris][:, :, 1] >= -1.0e-12, axis=1)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _warn_if_inverted_open_shell(
            verts, tris[keep], tags[keep]
        )
