from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .result import MeshInfo

logger = logging.getLogger(__name__)

# Matches hornlab-metal-bem's cut-plane snap tolerance so both solvers agree
# on what counts as "on a symmetry plane".
_SYMMETRY_SNAP_TOLERANCE = 1.0e-6


@dataclass
class LoadedMesh:
    grid: object  # bempp.api.Grid
    physical_tags: NDArray[np.int32]
    info: MeshInfo


class MeshError(Exception):
    pass


def load_mesh(
    path: str | Path,
    scale: float = 1.0,
    validate: bool = True,
    merge_tol: float = 1e-9,
    repair_normals: bool = False,
    require_closed: bool = False,
    native_symmetry_plane: str | None = None,
) -> LoadedMesh:
    """Load a .msh file into a bempp Grid with physical group tags.

    Gmsh/ABEC surface meshes can contain duplicate seam vertices. Bempp treats
    those as disconnected components unless we stitch them before grid creation.

    Canonical HornLab meshes are expected to arrive with outward-oriented
    triangle winding. Set ``repair_normals=True`` only for explicit
    compatibility with arbitrary external meshes that may use inward winding.
    """
    import bempp_cl.api as bempp_api
    import meshio

    path = Path(path)
    if not path.exists():
        raise MeshError(f"Mesh file not found: {path}")

    mesh = meshio.read(path)
    triangles = np.asarray(mesh.get_cells_type("triangle"), dtype=np.int32)
    if not triangles.size:
        raise MeshError("No triangles found in mesh")
    verts = np.asarray(mesh.points, dtype=np.float64) * scale
    try:
        phys_tags = np.asarray(
            mesh.get_cell_data("gmsh:physical", "triangle"),
            dtype=np.int32,
        )
    except (KeyError, ValueError) as exc:
        raise MeshError("Mesh file has no triangle physical-group tags") from exc
    phys_group_names = _extract_physical_names(path)
    for name, raw in getattr(mesh, "field_data", {}).items():
        values = np.asarray(raw).reshape(-1)
        if values.size >= 2 and int(values[1]) == 2:
            phys_group_names[int(values[0])] = str(name)

    verts, triangles, merged_vertices = _merge_duplicate_vertices(
        verts, triangles, merge_tol,
    )
    if merged_vertices:
        logger.info("Merged %d duplicate seam vertices", merged_vertices)

    # Remove degenerate triangles, including any created by seam merging.
    valid = ~(
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 0] == triangles[:, 2])
    )
    n_degen = np.sum(~valid)
    if n_degen > 0:
        logger.info("Removed %d degenerate triangles", n_degen)
        triangles = triangles[valid]
        phys_tags = phys_tags[valid]

    edge_incidence = (
        _edge_incidence_counts(triangles) if require_closed or validate else None
    )
    if require_closed:
        _require_closed_surface(
            verts, triangles, edge_incidence=edge_incidence,
        )
    elif validate and native_symmetry_plane is None:
        _warn_if_reduced_symmetry_mesh(
            verts, triangles, edge_incidence=edge_incidence,
        )
    elif validate:
        _check_declared_symmetry_plane(
            verts,
            triangles,
            native_symmetry_plane,
            edge_incidence=edge_incidence,
        )

    if validate:
        _validate_outward_normals(
            verts,
            triangles,
            repair=repair_normals,
            edge_incidence=edge_incidence,
        )
        _validate_physical_groups(phys_tags)
        _warn_if_inverted_open_shell(
            verts, triangles, phys_tags, edge_incidence=edge_incidence,
        )

    grid = bempp_api.Grid(verts.T, triangles.T.astype(np.int32), phys_tags)

    info = MeshInfo(
        n_vertices=len(verts),
        n_triangles=len(triangles),
        physical_groups=phys_group_names,
        bounding_box_m=(verts.min(axis=0), verts.max(axis=0)),
    )

    logger.info(
        "Loaded mesh: %d verts, %d tris, groups=%s",
        info.n_vertices, info.n_triangles, info.physical_groups,
    )

    return LoadedMesh(grid=grid, physical_tags=phys_tags, info=info)


def _extract_physical_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    in_block = False
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.strip()
                if line == "$PhysicalNames":
                    in_block = True
                    continue
                if line == "$EndPhysicalNames":
                    break
                if not in_block:
                    continue
                parts = line.split(maxsplit=2)
                if len(parts) < 3 or not parts[0].isdigit():
                    continue
                dim = int(parts[0])
                tag = int(parts[1])
                if dim == 2:
                    names[tag] = parts[2].strip().strip('"')
    except OSError:
        return names
    return names


def _merge_duplicate_vertices(
    verts: NDArray[np.float64],
    tris: NDArray[np.int32],
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.int32], int]:
    """Merge seam vertices within the requested Euclidean tolerance.

    Pairs come from a spatial tree, then union only after an exact
    squared-distance check. This preserves the Euclidean tolerance semantics
    at search-cell boundaries without making every mesh load Python-loop bound.
    """
    if tol <= 0 or len(verts) == 0:
        return verts, tris, 0

    if not np.isfinite(tol):
        raise MeshError("merge_tol must be finite")

    parent = np.arange(len(verts), dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    tol_sq = float(tol) ** 2
    # ``query_pairs(tol)`` can exclude a pair whose squared distance rounds to
    # exactly ``tol_sq``. Search one representable step wider, then retain the
    # squared-distance predicate as the semantic authority.
    search_radius = np.nextafter(float(tol), np.inf)
    pairs = cKDTree(verts).query_pairs(search_radius, output_type="ndarray")
    for left, right in pairs:
        left_index = int(left)
        right_index = int(right)
        delta = verts[right_index] - verts[left_index]
        if float(delta @ delta) > tol_sq:
            continue
        root_left = find(left_index)
        root_right = find(right_index)
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

    merged_verts = verts[unique_roots]
    merged_tris = inverse[tris].astype(np.int32, copy=False)
    return merged_verts, merged_tris, len(verts) - len(merged_verts)


def _validate_outward_normals(
    verts: NDArray[np.float64],
    tris: NDArray[np.int32],
    *,
    repair: bool = False,
    edge_incidence: tuple[NDArray[np.int32], NDArray[np.int64]] | None = None,
) -> None:
    """Validate outward winding, optionally repairing legacy external meshes."""
    if not _is_closed_two_manifold(tris, edge_incidence=edge_incidence):
        # Signed volume is origin-dependent for an open surface. Bare horns are
        # supported inputs, so do not reject or flip them solely because they
        # were translated across the coordinate origin.
        return

    signed_vol = _signed_mesh_volume_indicator(verts, tris)
    if signed_vol >= 0:
        return

    if repair:
        logger.info("Flipping triangle winding (signed volume negative)")
        tris[:, [1, 2]] = tris[:, [2, 1]]
        return

    raise MeshError(
        "Mesh triangle winding appears inward (signed volume negative). "
        "Canonical meshes must be emitted with outward normals by the mesher; "
        "pass repair_normals=True only for explicit external-mesh compatibility."
    )


def _signed_mesh_volume_indicator(
    verts: NDArray[np.float64],
    tris: NDArray[np.int32],
) -> float:
    """Return the signed volume indicator used for closed-surface winding."""
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    return float(np.sum(p0 * np.cross(p1, p2)))


def _is_closed_two_manifold(
    triangles_nx3: NDArray[np.int32],
    *,
    edge_incidence: tuple[NDArray[np.int32], NDArray[np.int64]] | None = None,
) -> bool:
    if edge_incidence is None:
        edge_incidence = _edge_incidence_counts(triangles_nx3)
    _edges, counts = edge_incidence
    return bool(counts.size and np.all(counts == 2))


def _validate_physical_groups(phys_tags: NDArray[np.int32]) -> None:
    unique = np.unique(phys_tags)
    if not np.any(unique >= 2):
        raise MeshError(
            f"No velocity source (tag >= 2) found. Tags: {unique.tolist()}"
        )
    if not np.any(unique == 1):
        logger.warning("No rigid wall (tag 1) in mesh")


def _validate_velocity_source_tags(
    physical_tags: NDArray[np.int32],
    velocity_sources: dict[int, float],
) -> None:
    """Require every configured velocity-source tag to exist in the mesh."""
    mesh_tags = {int(tag) for tag in np.unique(physical_tags)}
    missing_tags = sorted(set(velocity_sources) - mesh_tags)
    if missing_tags:
        raise ValueError(
            f"velocity_sources tags {missing_tags} are not present in the mesh; "
            f"available physical tags: {sorted(mesh_tags)}"
        )


def open_boundary_edges(
    triangles_nx3: NDArray[np.int32],
    *,
    edge_incidence: tuple[NDArray[np.int32], NDArray[np.int64]] | None = None,
) -> NDArray[np.int32]:
    """Return ``(n, 2)`` sorted vertex pairs for edges used by exactly one triangle.

    A closed surface has no open boundary edges; a mirror-reduced mesh has its
    open rim on the cut plane(s). Ported from hornlab-metal-bem so both
    solvers share the canonical-mesh closure contract.
    """
    if edge_incidence is None:
        edge_incidence = _edge_incidence_counts(triangles_nx3)
    unique_edges, counts = edge_incidence
    if unique_edges.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.ascontiguousarray(unique_edges[counts == 1], dtype=np.int32)


def _edge_incidence_counts(
    triangles_nx3: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int64]]:
    tris = np.asarray(triangles_nx3)
    if tris.size == 0:
        return np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.int64)
    edges = np.sort(
        np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]])),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return (
        np.ascontiguousarray(unique_edges, dtype=np.int32),
        np.ascontiguousarray(counts, dtype=np.int64),
    )


def _require_closed_surface(
    vertices_nx3: NDArray[np.float64],
    triangles_nx3: NDArray[np.int32],
    *,
    edge_incidence: tuple[NDArray[np.int32], NDArray[np.int64]] | None = None,
) -> None:
    """Raise when a closed-mode caller gives this backend an open surface."""
    verts = np.asarray(vertices_nx3, dtype=np.float64)
    if edge_incidence is None:
        edge_incidence = _edge_incidence_counts(
            np.asarray(triangles_nx3, dtype=np.int32)
        )
    edges, counts = edge_incidence
    bad = counts != 2
    if not np.any(bad):
        return
    bad_edges = edges[bad]
    open_count = int(np.count_nonzero(counts == 1))
    nonmanifold_count = int(np.count_nonzero(counts > 2))
    parts: list[str] = []
    if open_count:
        parts.append(f"{open_count} open boundary edges")
    if nonmanifold_count:
        parts.append(f"{nonmanifold_count} non-manifold edges")
    if not parts:
        parts.append(f"{bad_edges.shape[0]} invalid edges")
    example = verts[bad_edges[0]].round(6).tolist()
    raise MeshError(
        f"Mesh has {' and '.join(parts)} but the caller requires a closed "
        f"2-manifold surface (require_closed=True). Example invalid edge "
        f"between vertices {example}. The box is leaking or non-manifold — "
        "regenerate the mesh."
    )


def detect_reduced_symmetry_plane(
    vertices_nx3: NDArray[np.float64],
    triangles_nx3: NDArray[np.int32],
    *,
    tolerance: float = _SYMMETRY_SNAP_TOLERANCE,
    edge_incidence: tuple[NDArray[np.int32], NDArray[np.int64]] | None = None,
) -> str | None:
    """Heuristically detect mirror-reduced meshes (quarter/half models).

    Conservative on purpose: only reports a candidate when the mesh lives on
    the positive side of a candidate plane, has a meaningful set of used
    vertices on that plane, and every open boundary edge is explained by the
    candidate plane set. Ported from hornlab-metal-bem.
    """
    vertices = np.asarray(vertices_nx3, dtype=np.float64)
    triangles = np.asarray(triangles_nx3, dtype=np.int32)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or triangles.size == 0:
        return None

    used_vertices = np.unique(triangles.reshape(-1))
    if used_vertices.size == 0:
        return None

    boundary_edges = open_boundary_edges(
        triangles, edge_incidence=edge_incidence,
    )
    if boundary_edges.size == 0:
        return None

    used = vertices[used_vertices]
    candidates: list[str] = []
    boundary_on_plane: dict[str, NDArray[np.bool_]] = {}
    for plane, component in (("yz", 0), ("xz", 1), ("xy", 2)):
        values = used[:, component]
        on_plane = np.abs(values) <= tolerance
        edge_values = vertices[boundary_edges, component]
        boundary_on_plane[plane] = np.all(
            np.abs(edge_values) <= tolerance, axis=1
        )
        has_positive_side = bool(np.max(values) > tolerance)
        meaningful_count = int(np.count_nonzero(on_plane))
        if (
            np.min(values) >= -tolerance
            and has_positive_side
            and meaningful_count >= 2
            and np.count_nonzero(boundary_on_plane[plane]) >= 2
        ):
            candidates.append(plane)

    if not candidates:
        return None

    for plane in candidates:
        if np.all(boundary_on_plane[plane]):
            return plane
    if (
        "yz" in candidates
        and "xz" in candidates
        and np.all(boundary_on_plane["yz"] | boundary_on_plane["xz"])
    ):
        return "yz+xz"
    return None


def _warn_if_reduced_symmetry_mesh(
    vertices_nx3: NDArray[np.float64],
    triangles_nx3: NDArray[np.int32],
    *,
    edge_incidence: tuple[NDArray[np.int32], NDArray[np.int64]] | None = None,
) -> None:
    suspected = detect_reduced_symmetry_plane(
        vertices_nx3,
        triangles_nx3,
        edge_incidence=edge_incidence,
    )
    if suspected is None:
        return
    warnings.warn(
        "Mesh looks like a mirror-reduced native-symmetry mesh "
        f"(suspected plane {suspected!r}), but native_symmetry_plane was not "
        "declared. Set that option so Bempp reconstructs the mirrored geometry, "
        "or mesh the full domain. If the rim is a real open boundary (bare "
        "horn), ignore this warning.",
        RuntimeWarning,
        stacklevel=3,
    )


def _plane_set(plane: str | None) -> frozenset[str]:
    return frozenset(str(plane).strip().lower().split("+")) if plane else frozenset()


def _check_declared_symmetry_plane(
    vertices_nx3: NDArray[np.float64],
    triangles_nx3: NDArray[np.int32],
    declared_plane: str,
    *,
    edge_incidence: tuple[NDArray[np.int32], NDArray[np.int64]] | None = None,
) -> None:
    """Cross-check a declared symmetry plane against the mesh's own geometry.

    Declaring a plane used to switch the reduced-mesh detector off entirely,
    which is exactly when it is most useful: it already returns the right
    answer, it was simply never consulted. A quarter mesh declared as a half
    model (``quadrants=12`` on a ``quadrants=1`` mesh) is mirrored in one axis
    only, leaving the other cut plane open, and the solve proceeds on a torn
    shell. Nothing downstream catches that -- the expansion's own edge-count
    invariant balances, because the unmirrored cut simply reads as a rim.
    """
    detected = detect_reduced_symmetry_plane(
        vertices_nx3, triangles_nx3, edge_incidence=edge_incidence,
    )
    if detected is None:
        return
    detected_set = _plane_set(detected)
    declared_set = _plane_set(declared_plane)
    if detected_set == declared_set:
        return
    if detected_set > declared_set:
        missing = sorted(detected_set - declared_set)
        raise MeshError(
            f"Mesh is cut on {sorted(detected_set)} but native_symmetry_plane "
            f"declares only {sorted(declared_set)}. The {missing} cut "
            "plane(s) would not be mirrored, so the solve would run on a torn "
            "shell. Declare the full reduction (e.g. 'yz+xz' for a quarter "
            "model) or supply a mesh matching the declared plane."
        )
    warnings.warn(
        f"native_symmetry_plane={declared_plane!r} was declared but the mesh "
        f"looks cut on {detected!r}. Mirroring will be attempted as declared; "
        "verify the mesh matches the requested symmetry domain.",
        RuntimeWarning,
        stacklevel=3,
    )


def open_shell_bore_alignment(
    vertices_nx3: NDArray[np.float64],
    triangles_nx3: NDArray[np.int32],
    phys_tags: NDArray[np.int32],
    *,
    wall_tag: int = 1,
    source_tag: int = 2,
    band_fraction: float = 0.2,
) -> float | None:
    """Return the near-throat wall area fraction whose normal faces the bore.

    A bare zero-thickness shell has no interior, so the signed-volume indicator
    cannot judge its winding -- and about any fixed material point it reports
    the opposite sign for rollback profiles, whose wall curls back past the
    mouth plane. The throat collar is the part that is well defined: there the
    wall is a flaring tube around the source cap, so a normal facing the
    acoustic domain is exactly one with a negative radial component about the
    cap's own axis. Both references are taken from the mesh, so the measure is
    invariant under translation, rotation and vertical offsets.
    Ported from hornlab-metal-bem.

    Returns ``None`` when the mesh has no wall/source pair or no measurable
    collar -- an open sheet with the source on its free rim, for instance.
    """
    tags = np.asarray(phys_tags, dtype=np.int32)
    tris = np.asarray(triangles_nx3, dtype=np.int64)
    verts = np.asarray(vertices_nx3, dtype=np.float64)
    wall_mask = tags == int(wall_tag)
    source_mask = tags == int(source_tag)
    if not np.any(wall_mask) or not np.any(source_mask):
        return None

    s0, s1, s2 = (verts[tris[source_mask, i]] for i in range(3))
    source_area_vectors = np.cross(s1 - s0, s2 - s0)
    source_areas = 0.5 * np.linalg.norm(source_area_vectors, axis=1)
    if not np.any(source_areas > 0.0):
        return None
    cap_centroid = np.average((s0 + s1 + s2) / 3.0, weights=source_areas, axis=0)
    axis = source_area_vectors.sum(axis=0)
    axis_length = float(np.linalg.norm(axis))
    if axis_length <= 1.0e-12:
        return None
    axis = axis / axis_length

    w0, w1, w2 = (verts[tris[wall_mask, i]] for i in range(3))
    wall_area_vectors = np.cross(w1 - w0, w2 - w0)
    wall_areas = 0.5 * np.linalg.norm(wall_area_vectors, axis=1)
    offsets = (w0 + w1 + w2) / 3.0 - cap_centroid
    axial = offsets @ axis
    radial = offsets - np.outer(axial, axis)
    radial_lengths = np.linalg.norm(radial, axis=1)

    span = float(axial.max() - axial.min())
    if span <= 0.0:
        return None
    band = axial <= axial.min() + float(band_fraction) * span
    band &= (radial_lengths > 1.0e-12) & (wall_areas > 0.0)
    if not np.any(band):
        return None

    inward = -radial[band] / radial_lengths[band, None]
    facing_bore = np.sum(wall_area_vectors[band] * inward, axis=1) > 0.0
    banded_areas = wall_areas[band]
    return float(np.sum(banded_areas[facing_bore]) / np.sum(banded_areas))


def _warn_if_inverted_open_shell(
    vertices_nx3: NDArray[np.float64],
    triangles_nx3: NDArray[np.int32],
    phys_tags: NDArray[np.int32],
    *,
    edge_incidence: tuple[NDArray[np.int32], NDArray[np.int64]] | None = None,
    tolerance: float = 0.9,
) -> None:
    """Warn when an open shell's wall and source cap disagree on the bore side.

    ``_validate_outward_normals`` deliberately declines to judge a non-closed
    mesh, and ``require_closed_mesh`` defaults to ``False`` because a bare horn
    is a supported configuration here. That leaves the one case where the
    winding is both unconstrained and acoustically decisive: a bare shell whose
    wall points away from the bore solves as an almost transparent sheet, with
    a throat impedance wrong by a factor of several. This is a warning rather
    than an error because the measure cannot judge every open mesh; the mesher
    owns the contract, and this only catches meshes that predate or bypass it.
    """
    if _is_closed_two_manifold(triangles_nx3, edge_incidence=edge_incidence):
        return
    # A mirror-reduced closed body is open only on its cut planes, and carries
    # outer walls whose normals correctly face away from the bore. Only a mesh
    # with a genuinely free rim is a candidate bare shell.
    if (
        detect_reduced_symmetry_plane(
            vertices_nx3, triangles_nx3, edge_incidence=edge_incidence
        )
        is not None
    ):
        return

    alignment = open_shell_bore_alignment(vertices_nx3, triangles_nx3, phys_tags)
    if alignment is None or alignment >= float(tolerance):
        return
    warnings.warn(
        "Open mesh wall normals disagree with the source cap about which side "
        f"is the acoustic domain: only {alignment:.3f} of the near-throat wall "
        "area faces the bore. A bare shell wound this way radiates close to a "
        "free monopole and its throat impedance is wrong by a large factor. "
        "Regenerate the mesh with a current hornlab-waveguide-mesher, which "
        "pins bare-shell winding from the builder parameterisation.",
        RuntimeWarning,
        stacklevel=3,
    )
