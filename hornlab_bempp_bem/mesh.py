from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

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
    tri_key = "triangle" if "triangle" in mesh.cells_dict else "triangle3"
    if tri_key not in mesh.cells_dict:
        raise MeshError("No triangles found in mesh")

    triangles = np.asarray(mesh.cells_dict[tri_key], dtype=np.int32)
    verts = np.asarray(mesh.points, dtype=np.float64) * scale
    phys_tags = _extract_physical_tags(mesh, tri_key)
    phys_group_names = _extract_physical_names(path)

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

    if validate:
        _validate_outward_normals(
            verts,
            triangles,
            repair=repair_normals,
        )
        _validate_physical_groups(phys_tags)
        boundary = open_boundary_edges(triangles)
        if boundary.size:
            if require_closed:
                # A closed-mode mesh (enclosure / capped free-standing box)
                # with open boundary edges is a leaking model: this backend
                # has no symmetry support, so unlike hornlab-metal-bem's
                # cut-plane guard there is no legitimate reason for open
                # edges here — the solve would run and produce silently
                # wrong physics.
                example = verts[boundary[0]].round(6).tolist()
                raise MeshError(
                    f"Mesh has {boundary.shape[0]} open boundary edges but the "
                    "caller requires a closed surface (require_closed=True). "
                    f"Example open edge between vertices {example}. The box is "
                    "leaking — regenerate the mesh."
                )
            _warn_if_reduced_symmetry_mesh(verts, triangles)

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


def _extract_physical_tags(mesh, tri_key: str) -> NDArray[np.int32]:
    for key, by_type in mesh.cell_data_dict.items():
        if "physical" in key and tri_key in by_type:
            return np.asarray(by_type[tri_key], dtype=np.int32)
    raise MeshError("Mesh file has no triangle physical-group tags")


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
    """Merge coincident seam vertices and remap triangle connectivity."""
    if tol <= 0 or len(verts) == 0:
        return verts, tris, 0

    keys = np.round(verts / tol).astype(np.int64)
    _, first_indices, inverse = np.unique(
        keys,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    if len(first_indices) == len(verts):
        return verts, tris, 0

    merged_verts = verts[first_indices]
    merged_tris = inverse[tris].astype(np.int32, copy=False)
    return merged_verts, merged_tris, len(verts) - len(merged_verts)


def _validate_outward_normals(
    verts: NDArray[np.float64],
    tris: NDArray[np.int32],
    *,
    repair: bool = False,
) -> None:
    """Validate outward winding, optionally repairing legacy external meshes."""
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


def _validate_physical_groups(phys_tags: NDArray[np.int32]) -> None:
    unique = np.unique(phys_tags)
    if not np.any(unique >= 2):
        raise MeshError(
            f"No velocity source (tag >= 2) found. Tags: {unique.tolist()}"
        )
    if not np.any(unique == 1):
        logger.warning("No rigid wall (tag 1) in mesh")


def open_boundary_edges(
    triangles_nx3: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Return ``(n, 2)`` sorted vertex pairs for edges used by exactly one triangle.

    A closed surface has no open boundary edges; a mirror-reduced mesh has its
    open rim on the cut plane(s). Ported from hornlab-metal-bem so both
    solvers share the canonical-mesh closure contract.
    """
    tris = np.asarray(triangles_nx3)
    if tris.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    edges = np.sort(
        np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]])),
        axis=1,
    )
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return np.ascontiguousarray(unique_edges[counts == 1], dtype=np.int32)


def detect_reduced_symmetry_plane(
    vertices_nx3: NDArray[np.float64],
    triangles_nx3: NDArray[np.int32],
    *,
    tolerance: float = _SYMMETRY_SNAP_TOLERANCE,
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

    boundary_edges = open_boundary_edges(triangles)
    if boundary_edges.size == 0:
        return None

    used = vertices[used_vertices]
    candidates: list[tuple[str, int]] = []
    for plane, component in (("yz", 0), ("xz", 1), ("xy", 2)):
        values = used[:, component]
        on_plane = np.abs(values) <= tolerance
        has_positive_side = bool(np.max(values) > tolerance)
        meaningful_count = int(np.count_nonzero(on_plane))
        if (
            np.min(values) >= -tolerance
            and has_positive_side
            and meaningful_count >= 2
            and _count_edges_on_plane(vertices, boundary_edges, component, tolerance) >= 2
        ):
            candidates.append((plane, component))

    if not candidates:
        return None

    candidate_components = {plane: component for plane, component in candidates}
    for plane, component in candidate_components.items():
        if _all_edges_on_any_plane(vertices, boundary_edges, [component], tolerance):
            return plane
    if (
        "yz" in candidate_components
        and "xz" in candidate_components
        and _all_edges_on_any_plane(
            vertices,
            boundary_edges,
            [candidate_components["yz"], candidate_components["xz"]],
            tolerance,
        )
    ):
        return "yz+xz"
    return None


def _warn_if_reduced_symmetry_mesh(
    vertices_nx3: NDArray[np.float64],
    triangles_nx3: NDArray[np.int32],
) -> None:
    suspected = detect_reduced_symmetry_plane(vertices_nx3, triangles_nx3)
    if suspected is None:
        return
    warnings.warn(
        "Mesh looks like a mirror-reduced native-symmetry mesh "
        f"(suspected plane {suspected!r}), but this backend has no symmetry "
        "support: it would silently solve the open shell instead of the "
        "mirrored geometry. Mesh the full domain for bempp, or solve the "
        f"reduced mesh with hornlab-metal-bem and native_symmetry_plane="
        f"{suspected!r}. If the rim is a real open boundary (bare horn), "
        "ignore this warning.",
        RuntimeWarning,
        stacklevel=3,
    )


def _count_edges_on_plane(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int32],
    component: int,
    tolerance: float,
) -> int:
    return int(
        np.count_nonzero(_edge_on_plane_mask(vertices, edges, component, tolerance))
    )


def _all_edges_on_any_plane(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int32],
    components: list[int],
    tolerance: float,
) -> bool:
    explained = np.zeros(edges.shape[0], dtype=bool)
    for component in components:
        explained |= _edge_on_plane_mask(vertices, edges, component, tolerance)
    return bool(np.all(explained))


def _edge_on_plane_mask(
    vertices: NDArray[np.float64],
    edges: NDArray[np.int32],
    component: int,
    tolerance: float,
) -> NDArray[np.bool_]:
    edge_values = vertices[edges, component]
    return np.all(np.abs(edge_values) <= tolerance, axis=1)
