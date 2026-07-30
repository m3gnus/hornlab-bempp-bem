"""Mirror-reduced half/quarter assembly for the Bempp backend.

The reduced mesh is expanded only as geometry. Boundary equations are tested
on the original image block, while trial coefficients are tied across their
mirror orbits. Dense assembly therefore scales with one reduced row block
instead of assembling and solving the complete mirrored system.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import logging
import time
from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from ._blas_threads import limited_blas_threads

logger = logging.getLogger(__name__)


_TRANSVERSE_PLANES = {"yz", "xz", "yz+xz"}


def _image_signs(symmetry_plane: str) -> tuple[tuple[int, int, int], ...]:
    plane = str(symmetry_plane).strip().lower()
    if plane not in _TRANSVERSE_PLANES:
        raise ValueError(
            "Bempp native symmetry currently supports 'yz', 'xz', and "
            "'yz+xz'"
        )
    x_signs = (1, -1) if "yz" in plane else (1,)
    y_signs = (1, -1) if "xz" in plane else (1,)
    return tuple((sx, sy, 1) for sx, sy in product(x_signs, y_signs))


def _coord_key(
    point: NDArray[np.floating],
    tolerance: float,
) -> tuple[int, int, int]:
    return tuple(
        np.round(np.asarray(point, dtype=np.float64) / tolerance)
        .astype(np.int64)
        .tolist()
    )


@dataclass(frozen=True)
class ExpandedSymmetryMesh:
    vertices_nx3: NDArray[np.float64]
    triangles_nx3: NDArray[np.int64]
    physical_tags: NDArray[np.int32]
    image_signs: tuple[tuple[int, int, int], ...]
    reduced_element_count: int
    # The tolerance actually used, so callers matching coordinates against the
    # expanded grid can snap their own the same way.
    plane_tolerance: float = 0.0


def _snap_to_planes(
    coordinates: NDArray[np.float64],
    symmetry_plane: str,
    plane_tolerance: float,
) -> NDArray[np.float64]:
    """Snap coordinates within ``plane_tolerance`` of a mirror plane to zero.

    Mutates and returns ``coordinates``. Doing this exactly, rather than
    relying on a rounding bucket, is what makes a seam weld onto its own mirror
    image: ``0.0 * -1`` is ``-0.0``, which compares and hashes equal to ``0.0``.
    """
    for _name, component in _plane_components(symmetry_plane):
        values = coordinates[:, component]
        values[np.abs(values) <= plane_tolerance] = 0.0
    return coordinates


def _plane_components(symmetry_plane: str) -> tuple[tuple[str, int], ...]:
    """Coordinate index mirrored by each plane named in ``symmetry_plane``."""
    plane = str(symmetry_plane).strip().lower()
    found = [("yz", 0)] if "yz" in plane else []
    if "xz" in plane:
        found.append(("xz", 1))
    return tuple(found)


def _triangle_areas(
    vertices: NDArray[np.float64],
    triangles: NDArray[np.int64],
) -> NDArray[np.float64]:
    if triangles.size == 0:
        return np.zeros(0, dtype=np.float64)
    p0 = vertices[triangles[:, 0]]
    edge1 = vertices[triangles[:, 1]] - p0
    edge2 = vertices[triangles[:, 2]] - p0
    return np.linalg.norm(np.cross(edge1, edge2), axis=1) / 2.0


def _min_triangle_area(
    vertices: NDArray[np.float64],
    triangles: NDArray[np.int64],
) -> float:
    """Smallest triangle area, or 0.0 for an empty mesh."""
    areas = _triangle_areas(vertices, triangles)
    return float(areas.min()) if areas.size else 0.0


def _degenerate_triangles(
    vertices: NDArray[np.float64],
    triangles: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Indices of triangles with zero area or a repeated vertex."""
    if triangles.size == 0:
        return np.zeros(0, dtype=np.int64)
    repeated = (
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 0] == triangles[:, 2])
    )
    return np.flatnonzero(repeated | (_triangle_areas(vertices, triangles) <= 0.0))


def _require_cut_boundary(
    vertices: NDArray[np.float64],
    triangles: NDArray[np.int64],
    symmetry_plane: str,
    plane_tolerance: float,
) -> None:
    """Every declared mirror plane must carry a real cut boundary.

    Touching the plane is not enough. A closed body tangent at a single vertex
    satisfies ``min(coord) == 0`` and mirrors into two shells joined at a point,
    which is not a manifold surface and which the open-edge count cannot see
    because neither shell has any open edge at all.
    """
    from .mesh import open_boundary_edges

    boundary = open_boundary_edges(np.asarray(triangles, dtype=np.int32))
    for name, component in _plane_components(symmetry_plane):
        if boundary.size:
            on_plane = np.all(
                np.abs(vertices[boundary, component]) <= plane_tolerance, axis=1,
            )
            if np.any(on_plane):
                continue
        raise ValueError(
            f"reduced mesh declares the {name!r} mirror plane but has no open "
            "boundary edge lying in it, so there is nothing for mirroring to "
            "weld. A body that only touches the plane mirrors into two pieces "
            "joined at a point. Check that the mesh is really cut on this "
            "plane."
        )


def _require_seamless_expansion(
    vertices: NDArray[np.float64],
    triangles: NDArray[np.int64],
    expanded_triangles: NDArray[np.int64],
    symmetry_plane: str,
    image_count: int,
    plane_tolerance: float,
) -> None:
    """Reject a reduced mesh whose cut boundary did not weld under mirroring.

    A reduced model is the positive-side piece of a mirror-symmetric body, so
    every open boundary edge lying in a mirror plane is a *cut* edge: mirroring
    must consume it against its own image. Edges elsewhere -- a horn mouth or
    throat rim -- are real openings and survive once per image.

    This is a backstop. Plane snapping in ``expand_symmetry_mesh`` welds any
    seam within ``plane_tolerance`` exactly, and the touch check rejects a mesh
    that misses the plane entirely, so between them a weld failure should be
    unreachable from geometry; what remains is dedup collisions and degenerate
    or non-manifold input. It does not detect a single seam vertex lifted past
    the tolerance while its neighbours stay put -- that edge simply reclassifies
    as a rim and the count still balances. Meshers do not produce that shape,
    but ``require_closed_mesh`` is the check that would catch it.
    """
    from .mesh import _edge_incidence_counts, open_boundary_edges

    reduced_open = open_boundary_edges(triangles.astype(np.int32))
    on_plane = np.zeros(reduced_open.shape[0], dtype=bool)
    for _name, component in _plane_components(symmetry_plane):
        if reduced_open.size:
            edge_values = vertices[reduced_open, component]
            on_plane |= np.all(np.abs(edge_values) <= plane_tolerance, axis=1)
    expected = image_count * int(np.count_nonzero(~on_plane))

    expanded_open = open_boundary_edges(
        np.asarray(expanded_triangles, dtype=np.int32),
        edge_incidence=_edge_incidence_counts(
            np.asarray(expanded_triangles, dtype=np.int32),
        ),
    )
    actual = int(expanded_open.shape[0])
    if actual == expected:
        return

    cut_edges = int(np.count_nonzero(on_plane))
    raise ValueError(
        f"mirroring a '{symmetry_plane}' reduced mesh left {actual} open "
        f"boundary edges but the geometry accounts for {expected} "
        f"({int(np.count_nonzero(~on_plane))} non-cut open edges x "
        f"{image_count} images; {cut_edges} cut edges should have welded). "
        "The reduced mesh's cut boundary does not meet the mirror plane "
        "within the seam tolerance, so the expanded model is torn or "
        "disjoint. Check that the mesh is snapped to the plane and carries "
        "no offset applied after the cut."
    )


def expand_symmetry_mesh(
    vertices_nx3: NDArray[np.floating],
    triangles_nx3: NDArray[np.integer],
    physical_tags: NDArray[np.integer],
    symmetry_plane: str,
    *,
    tolerance: float = 1.0e-9,
    plane_tolerance: float | None = None,
    validate_seam: bool = True,
) -> ExpandedSymmetryMesh:
    """Mirror a positive-side reduced mesh while sharing seam vertices.

    ``tolerance`` is the vertex-deduplication quantization bucket. ``plane_
    tolerance`` (default ``mesh._SYMMETRY_SNAP_TOLERANCE``) is the distance
    within which a vertex counts as lying *on* a mirror plane; such vertices
    are snapped to exactly zero before mirroring so a seam welds exactly rather
    than depending on both sides landing in the same quantization bucket.
    Keeping the two separate is what lets the seam tolerance be a physical
    length while dedup stays fine enough not to merge distinct vertices. Grids
    reaching here through ``load_mesh`` are already scaled to metres, so the
    default is 1 um of real space; a caller passing raw coordinates is
    responsible for a tolerance in those units.

    ``validate_seam`` gates the *geometric preconditions* -- that the mesh
    reaches each declared plane, carries a real cut boundary on it, and mirrors
    without tearing. Turning it off is for diagnostics and for the deliberate
    mirrored-pair case; the checks that keep Bempp's input valid at all (which
    side of the plane the mesh lies on, and degenerate elements) always run.
    """
    from .mesh import _SYMMETRY_SNAP_TOLERANCE

    vertices = np.array(vertices_nx3, dtype=np.float64, copy=True)
    triangles = np.asarray(triangles_nx3, dtype=np.int64)
    tags = np.asarray(physical_tags, dtype=np.int32)
    signs = _image_signs(symmetry_plane)
    if plane_tolerance is None:
        plane_tolerance = _SYMMETRY_SNAP_TOLERANCE

    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_nx3 must have shape (n_vertices, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles_nx3 must have shape (n_triangles, 3)")
    if tags.shape != (triangles.shape[0],):
        raise ValueError("physical_tags must have one value per triangle")

    plane = str(symmetry_plane).strip().lower()
    axis_names = {0: "X", 1: "Y"}
    # Unused vertices say nothing about the geometry being solved, and letting
    # them satisfy the plane checks is a way to smuggle an offset mesh through.
    used = np.unique(triangles.reshape(-1))
    for name, component in _plane_components(plane):
        values = vertices[used, component]
        axis = axis_names[component]
        if float(values.min()) < -plane_tolerance:
            raise ValueError(
                f"{name} half/quarter mesh must lie in {axis} >= 0"
            )
        # A reduced mesh that never reaches its own mirror plane is not the
        # positive-side half of anything: mirroring it produces two separated
        # bodies. Catch it here rather than solving a disjoint model.
        #
        # This does rule out a legitimate but unrelated use of the image
        # method -- one body wholly on the positive side standing in for a
        # mirrored *pair* of bodies. That is not what this package's reduced
        # meshes mean, and allowing it would also let through the mesher's
        # VerticalOffset bug, so it is refused by default; validate_seam=False
        # is the escape hatch for a caller who really wants it.
        if validate_seam and float(values.min()) > plane_tolerance:
            raise ValueError(
                f"{name} half/quarter mesh does not touch its mirror plane "
                f"(minimum {axis} is {float(values.min()):.6g} m, seam "
                f"tolerance {plane_tolerance:.6g} m). Mirroring it would "
                "produce two disjoint shells. The mesh is probably offset "
                "away from the cut plane; pass validate_seam=False if a "
                "mirrored pair of separate bodies is genuinely intended."
            )

    before_area = _min_triangle_area(vertices, triangles)
    _snap_to_planes(vertices, plane, plane_tolerance)

    # Moving a vertex by at most plane_tolerance cannot flatten a healthy
    # element, but it can merge two vertices that were already within a
    # tolerance of the plane and of each other -- a thin feature normal to the
    # cut. Check absolutely rather than relatively so a mesh that already
    # contained a degenerate triangle cannot mask the new one.
    degenerate = _degenerate_triangles(vertices, triangles)
    if degenerate.size:
        raise ValueError(
            f"snapping the seam to the {plane!r} mirror plane(s) collapsed "
            f"{degenerate.size} triangle(s) to zero area (smallest area before "
            f"snapping was {before_area:.3e} m^2); first collapsed triangle is "
            f"{triangles[degenerate[0]].tolist()}. A feature thinner than the "
            f"seam tolerance ({plane_tolerance:.3g} m) cannot be distinguished "
            "from the cut plane; fix the mesh or lower plane_tolerance."
        )

    # The cut boundary is what mirroring consumes. A reduced model that merely
    # grazes the plane -- a closed body tangent at a single vertex, or one whose
    # seam vertices have been lifted off it one at a time -- mirrors into pieces
    # joined at a point or torn along the cut. Require a real cut edge on each
    # declared plane. Gated with the edge-count invariant so the diagnostics
    # script can still expand a suspect mesh to look at it.
    if validate_seam:
        _require_cut_boundary(vertices, triangles, plane, plane_tolerance)

    vertex_lookup: dict[tuple[int, int, int], int] = {}
    expanded_vertices: list[NDArray[np.float64]] = []
    image_vertex_maps: list[NDArray[np.int64]] = []
    for sign in signs:
        sign_array = np.asarray(sign, dtype=np.float64)
        remap = np.empty(vertices.shape[0], dtype=np.int64)
        for index, point in enumerate(vertices):
            mirrored = point * sign_array
            key = _coord_key(mirrored, tolerance)
            mapped = vertex_lookup.get(key)
            if mapped is None:
                mapped = len(expanded_vertices)
                vertex_lookup[key] = mapped
                expanded_vertices.append(mirrored)
            remap[index] = mapped
        image_vertex_maps.append(remap)

    element_blocks: list[NDArray[np.int64]] = []
    for sign, remap in zip(signs, image_vertex_maps, strict=True):
        block = remap[triangles]
        if int(np.prod(sign)) < 0:
            block = block[:, [0, 2, 1]]
        element_blocks.append(block)

    expanded_triangles = np.vstack(element_blocks)
    if validate_seam:
        _require_seamless_expansion(
            vertices,
            triangles,
            expanded_triangles,
            plane,
            len(signs),
            plane_tolerance,
        )

    return ExpandedSymmetryMesh(
        vertices_nx3=np.asarray(expanded_vertices, dtype=np.float64),
        triangles_nx3=expanded_triangles,
        physical_tags=np.tile(tags, len(signs)).astype(np.int32),
        image_signs=signs,
        reduced_element_count=triangles.shape[0],
        plane_tolerance=float(plane_tolerance),
    )


def _space_dof_coordinates(space) -> NDArray[np.float64]:
    """Return coordinates for every active scalar P1 dof in a space."""
    grid = space.grid
    vertices = np.asarray(grid.vertices, dtype=np.float64).T
    elements = np.asarray(grid.elements, dtype=np.int64).T
    local2global = np.asarray(space.local2global, dtype=np.int64)
    multipliers = np.asarray(space.local_multipliers)
    coordinates = np.full(
        (int(space.global_dof_count), 3), np.nan, dtype=np.float64,
    )
    for element_index in np.asarray(space.support_elements, dtype=np.int64):
        for local_index in range(3):
            if multipliers[element_index, local_index] == 0:
                continue
            dof = int(local2global[element_index, local_index])
            point = vertices[elements[element_index, local_index]]
            if np.isnan(coordinates[dof, 0]):
                coordinates[dof] = point
            elif np.linalg.norm(coordinates[dof] - point) > 1.0e-8:
                raise ValueError(f"P1 dof {dof} has inconsistent coordinates")
    if np.isnan(coordinates).any():
        raise ValueError("could not resolve every P1 dof coordinate")
    return coordinates


def _unique_coordinate_map(
    coordinates: NDArray[np.float64],
    tolerance: float,
) -> dict[tuple[int, int, int], int]:
    result: dict[tuple[int, int, int], int] = {}
    for index, point in enumerate(coordinates):
        key = _coord_key(point, tolerance)
        if key in result:
            raise ValueError(f"duplicate P1 coordinate near {point.tolist()}")
        result[key] = index
    return result


@dataclass
class SymmetryContext:
    """Spaces and sparse orbit maps reused by every sweep frequency."""

    symmetry_plane: str
    image_signs: tuple[tuple[int, int, int], ...]
    reduced_grid: object
    reduced_p1: object
    reduced_dp0: object
    expanded_grid: object
    full_p1: object
    full_dp0: object
    test_p1: object
    active_reduced_dofs: NDArray[np.int64]
    active_test_rows: NDArray[np.int64]
    pressure_expansion: sparse.csr_matrix
    reduced_mass_matrix: sparse.csr_matrix
    reduced_element_count: int

    @property
    def image_count(self) -> int:
        return len(self.image_signs)

    @property
    def active_dof_count(self) -> int:
        return len(self.active_reduced_dofs)

    def expand_neumann_coefficients(
        self,
        reduced_coefficients: NDArray[np.complexfloating],
    ) -> NDArray:
        coefficients = np.asarray(reduced_coefficients)
        if coefficients.shape != (self.reduced_dp0.global_dof_count,):
            raise ValueError("reduced Neumann coefficient count does not match DP0")
        return np.tile(coefficients, self.image_count)

    def wrap_reduced_pressure(self, active_coefficients: NDArray):
        import bempp_cl.api as bempp_api

        coefficients = np.zeros(
            self.reduced_p1.global_dof_count,
            dtype=np.asarray(active_coefficients).dtype,
        )
        coefficients[self.active_reduced_dofs] = active_coefficients
        return bempp_api.GridFunction(
            self.reduced_p1, coefficients=coefficients,
        )

    def wrap_expanded_pressure(self, active_coefficients: NDArray):
        import bempp_cl.api as bempp_api

        coefficients = self.pressure_expansion @ np.asarray(active_coefficients)
        return bempp_api.GridFunction(
            self.full_p1, coefficients=np.asarray(coefficients),
        )


def build_symmetry_context(
    grid,
    physical_tags: NDArray[np.integer],
    symmetry_plane: str,
    *,
    tolerance: float = 1.0e-8,
    plane_tolerance: float | None = None,
) -> SymmetryContext:
    """Build the expanded geometry and even-pressure orbit maps.

    ``tolerance`` matches P1 dofs to their mirror images. ``plane_tolerance``
    decides what counts as lying on a mirror plane; see ``expand_symmetry_mesh``.
    """
    import bempp_cl.api as bempp_api

    expanded = expand_symmetry_mesh(
        np.asarray(grid.vertices, dtype=np.float64).T,
        np.asarray(grid.elements, dtype=np.int64).T,
        physical_tags,
        symmetry_plane,
        # Dedup finer than the dof-matching tolerance: distinct vertices must
        # never collide here. Seam welding does not depend on this bucket --
        # expand_symmetry_mesh snaps on-plane coordinates to exactly zero.
        tolerance=min(tolerance, 1.0e-9),
        plane_tolerance=plane_tolerance,
    )
    expanded_grid = bempp_api.Grid(
        expanded.vertices_nx3.T,
        expanded.triangles_nx3.T.astype(np.uint32),
    )

    # The output space includes cut-plane dofs. Real open-boundary dofs that
    # the full Bempp P1 space excludes are retained only as zero placeholders.
    reduced_p1 = bempp_api.function_space(
        grid, "P", 1, include_boundary_dofs=True,
    )
    reduced_dp0 = bempp_api.function_space(grid, "DP", 0)
    full_p1 = bempp_api.function_space(expanded_grid, "P", 1)
    full_dp0 = bempp_api.function_space(expanded_grid, "DP", 0)
    base_elements = np.arange(
        expanded.reduced_element_count, dtype=np.uint32,
    )
    test_p1 = bempp_api.function_space(
        expanded_grid,
        "P",
        1,
        support_elements=base_elements,
        include_boundary_dofs=True,
    )

    reduced_coords = _space_dof_coordinates(reduced_p1)
    # The expanded grid was built from snapped coordinates, so the reduced dof
    # coordinates must be snapped identically before they are matched against
    # it. Without this the orbit lookup misses by up to plane_tolerance and
    # raises "missing full dofs" -- the advertised seam tolerance would work in
    # expand_symmetry_mesh and then fail here, which is worse than not having
    # it, because it turns a repaired mesh into a hard error.
    _snap_to_planes(
        reduced_coords,
        symmetry_plane,
        expanded.plane_tolerance,
    )
    full_coords = _space_dof_coordinates(full_p1)
    test_coords = _space_dof_coordinates(test_p1)
    full_lookup = _unique_coordinate_map(full_coords, tolerance)
    test_lookup = _unique_coordinate_map(test_coords, tolerance)

    active_reduced: list[int] = []
    active_rows: list[int] = []
    orbit_rows: list[int] = []
    orbit_cols: list[int] = []
    covered_full_dofs: set[int] = set()
    for reduced_dof, point in enumerate(reduced_coords):
        original_key = _coord_key(point, tolerance)
        if original_key not in full_lookup:
            # A real free-boundary P1 dof is absent from the full space.
            continue
        test_row = test_lookup.get(original_key)
        if test_row is None:
            raise ValueError("reduced P1 dof is missing from symmetry test space")
        column = len(active_reduced)
        members: set[int] = set()
        for sign in expanded.image_signs:
            mirrored = point * np.asarray(sign, dtype=np.float64)
            full_dof = full_lookup.get(_coord_key(mirrored, tolerance))
            if full_dof is not None:
                members.add(full_dof)
        if not members:
            raise ValueError("empty mirror orbit for active reduced P1 dof")
        active_reduced.append(reduced_dof)
        active_rows.append(test_row)
        for full_dof in sorted(members):
            orbit_rows.append(full_dof)
            orbit_cols.append(column)
            covered_full_dofs.add(full_dof)

    # Structural invariant on the orbit map, NOT a mesh-validity check: the
    # expanded grid is built by mirroring, so every full dof is by construction
    # the image of some reduced vertex and this cannot fire for a geometrically
    # wrong mesh. Seam and plane validation lives in expand_symmetry_mesh.
    if covered_full_dofs != set(range(full_p1.global_dof_count)):
        missing = sorted(set(range(full_p1.global_dof_count)) - covered_full_dofs)
        raise ValueError(
            "reduced mesh does not cover every mirrored P1 orbit; "
            f"missing full dofs {missing[:8]}"
        )

    expansion = sparse.coo_matrix(
        (
            np.ones(len(orbit_rows), dtype=np.float64),
            (orbit_rows, orbit_cols),
        ),
        shape=(full_p1.global_dof_count, len(active_reduced)),
    ).tocsr()

    active_index = np.full(reduced_p1.global_dof_count, -1, dtype=np.int64)
    active_index[np.asarray(active_reduced, dtype=np.int64)] = np.arange(
        len(active_reduced), dtype=np.int64,
    )
    local2global = np.asarray(reduced_p1.local2global, dtype=np.int64)
    multipliers = np.asarray(reduced_p1.local_multipliers, dtype=np.float64)
    areas = np.asarray(grid.volumes, dtype=np.float64)
    mass_rows: list[int] = []
    mass_cols: list[int] = []
    mass_values: list[float] = []
    reference_mass = np.ones((3, 3), dtype=np.float64) + np.eye(3)
    for element_index in range(expanded.reduced_element_count):
        dofs = local2global[element_index]
        active = active_index[dofs]
        local_mass = (
            areas[element_index]
            / 12.0
            * reference_mass
            * np.outer(
                multipliers[element_index],
                multipliers[element_index],
            )
        )
        for local_row, row in enumerate(active):
            if row < 0:
                continue
            for local_col, col in enumerate(active):
                if col < 0:
                    continue
                mass_rows.append(int(row))
                mass_cols.append(int(col))
                mass_values.append(float(local_mass[local_row, local_col]))
    reduced_mass = sparse.coo_matrix(
        (mass_values, (mass_rows, mass_cols)),
        shape=(len(active_reduced), len(active_reduced)),
    ).tocsr()

    return SymmetryContext(
        symmetry_plane=str(symmetry_plane).strip().lower(),
        image_signs=expanded.image_signs,
        reduced_grid=grid,
        reduced_p1=reduced_p1,
        reduced_dp0=reduced_dp0,
        expanded_grid=expanded_grid,
        full_p1=full_p1,
        full_dp0=full_dp0,
        test_p1=test_p1,
        active_reduced_dofs=np.asarray(active_reduced, dtype=np.int64),
        active_test_rows=np.asarray(active_rows, dtype=np.int64),
        pressure_expansion=expansion,
        reduced_mass_matrix=reduced_mass,
        reduced_element_count=expanded.reduced_element_count,
    )


def dense_matrix(operator) -> NDArray:
    """Materialize a Bempp weak form as a NumPy dense array."""
    import bempp_cl.api as bempp_api

    matrix = bempp_api.as_matrix(operator.weak_form())
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    return np.asarray(matrix)


def reduce_trial_matrix(
    full_trial_matrix: NDArray,
    context: SymmetryContext,
) -> NDArray:
    """Select the real-image equations and tie mirror-orbit columns."""
    selected = np.asarray(full_trial_matrix)[context.active_test_rows]
    return np.asarray(selected @ context.pressure_expansion)


def reduce_test_vector(
    test_vector: NDArray,
    context: SymmetryContext,
) -> NDArray:
    return np.asarray(test_vector)[context.active_test_rows]


def _restricted_full_neumann(
    context: SymmetryContext,
    reduced_coefficients: NDArray[np.complexfloating],
    restrict: bool = True,
):
    """Expand an even Neumann trace and retain only its nonzero DP0 support."""
    import bempp_cl.api as bempp_api

    full_coefficients = context.expand_neumann_coefficients(
        reduced_coefficients,
    )
    support = np.flatnonzero(full_coefficients != 0)
    if restrict and 0 < len(support) < len(full_coefficients):
        space = bempp_api.function_space(
            context.expanded_grid,
            "DP",
            0,
            support_elements=support,
        )
        return bempp_api.GridFunction(
            space, coefficients=full_coefficients[support],
        )
    return bempp_api.GridFunction(
        context.full_dp0, coefficients=full_coefficients,
    )


def _gmres_outer_cycles(max_iter: int, restart: int) -> int:
    """Convert an inner-iteration budget into scipy GMRES restart cycles.

    ``scipy.sparse.linalg.gmres`` counts ``maxiter`` in *outer* cycles, each of
    up to ``restart`` inner iterations, so passing an inner-iteration budget
    straight through permits ``restart * max_iter`` iterations -- 500,000 at
    this package's defaults of 5000 and 100.

    Rounding *up* rather than down: scipy cannot stop mid-cycle, so a budget
    that is not a whole number of cycles has to either overshoot by less than
    one cycle or truncate. Truncating loses real solves -- a system needing 148
    iterations under ``gmres_max_iter=150`` would be cut off at 100 and return
    an unconverged answer -- while overshooting costs at most ``restart - 1``
    extra iterations. The default 5000 with restart 100 is exact either way.
    """
    max_iter = max(0, int(max_iter))
    restart = max(1, int(restart))
    return max(1, -(-max_iter // restart))


def assemble_and_solve_symmetry(
    context: SymmetryContext,
    reduced_neumann,
    k: complex,
    config,
    operator_kwargs: dict,
):
    """Assemble and solve the even mirror-reduced standard Neumann BIE."""
    import bempp_cl.api as bempp_api
    import scipy.linalg
    import scipy.sparse.linalg

    from .config import BIEFormulation, LinearSolver

    if config.formulation is BIEFormulation.BURTON_MILLER:
        raise NotImplementedError(
            "native half/quarter symmetry currently supports STANDARD and "
            "COMPLEX_K formulations"
        )
    if config.impedance_sources:
        raise NotImplementedError(
            "native half/quarter symmetry with Robin impedance boundaries is "
            "not implemented yet"
        )

    timings: dict[str, float] = {}
    started = time.perf_counter()
    full_neumann = _restricted_full_neumann(
        context,
        np.asarray(reduced_neumann.coefficients),
        restrict=getattr(config, "restrict_neumann_space", True),
    )
    timings["neumann_expansion_s"] = time.perf_counter() - started

    started = time.perf_counter()
    dlp = bempp_api.operators.boundary.helmholtz.double_layer(
        context.full_p1,
        context.test_p1,
        context.test_p1,
        k,
        **operator_kwargs,
    )
    slp = bempp_api.operators.boundary.helmholtz.single_layer(
        full_neumann.space,
        context.test_p1,
        context.test_p1,
        k,
        **operator_kwargs,
    )
    timings["operator_construction_s"] = time.perf_counter() - started

    started = time.perf_counter()
    slp_matrix = dense_matrix(slp)
    timings["slp_assembly_s"] = time.perf_counter() - started
    started = time.perf_counter()
    dlp_matrix = dense_matrix(dlp)
    timings["dlp_assembly_s"] = time.perf_counter() - started
    started = time.perf_counter()
    reduced_mass = context.reduced_mass_matrix
    lhs = (
        reduce_trial_matrix(dlp_matrix, context)
        - 0.5 * reduced_mass.toarray()
    )
    rhs = reduce_test_vector(
        slp_matrix @ np.asarray(full_neumann.coefficients),
        context,
    )
    timings["symmetry_reduction_s"] = time.perf_counter() - started

    solver_choice = config.solver
    if solver_choice is LinearSolver.AUTO:
        solver_choice = (
            LinearSolver.LU
            if context.reduced_element_count <= config.lu_threshold
            else LinearSolver.GMRES
        )

    iterations = None
    converged = True
    started = time.perf_counter()
    # LAPACK LU and the GMRES inner products both lose badly to BLAS threading
    # at these matrix sizes, while the dense reduction above genuinely wants
    # the threads -- so the limit is scoped to the solve. See _blas_threads.
    with limited_blas_threads(1):
        if solver_choice is LinearSolver.LU:
            active_pressure = scipy.linalg.solve(lhs, rhs)
        else:
            # Bempp's strong form is M^-1 A. Apply the same mass preconditioner
            # explicitly because the orbit-reduced matrix is no longer a native
            # BoundaryOperator object.
            mass_solver = scipy.sparse.linalg.splu(
                reduced_mass.astype(lhs.dtype).tocsc(),
            )
            strong_lhs = mass_solver.solve(lhs)
            strong_rhs = mass_solver.solve(rhs)
            counter = [0]

            def count_iteration(_residual):
                counter[0] += 1

            # A larger restart is materially more robust for the orbit-reduced
            # half system; ``None`` remains the public "automatic" setting.
            restart = (
                100 if config.gmres_restart is None else config.gmres_restart
            )
            outer_cycles = _gmres_outer_cycles(config.gmres_max_iter, restart)
            # No frequency continuation (x0 from the previous frequency): it was
            # measured at only 4% fewer iterations over a 24-frequency sweep,
            # and it made individual high frequencies *worse* -- 47 -> 48,
            # 46 -> 51 -- because a guess that is wrong in a different Krylov
            # subspace can beat starting from zero. Not worth the state.
            active_pressure, info = scipy.sparse.linalg.gmres(
                strong_lhs,
                strong_rhs,
                rtol=config.gmres_tol,
                atol=0.0,
                restart=restart,
                maxiter=outer_cycles,
                callback=count_iteration,
                callback_type="pr_norm",
            )
            iterations = counter[0]
            converged = info == 0
            if not converged:
                # The non-symmetric path warns here; without this a diverged
                # frequency is only visible as a `converged` flag on the result.
                logger.warning(
                    "symmetry GMRES did not converge (info=%d) at k=%.4f after "
                    "%d iterations (restart %d, tol %.1e)",
                    info, float(np.real(k)), iterations, restart,
                    config.gmres_tol,
                )
    timings["linear_solve_s"] = time.perf_counter() - started

    started = time.perf_counter()
    reduced_pressure = context.wrap_reduced_pressure(active_pressure)
    expanded_pressure = context.wrap_expanded_pressure(active_pressure)
    timings["solution_wrapping_s"] = time.perf_counter() - started
    return (
        reduced_pressure,
        expanded_pressure,
        full_neumann,
        iterations,
        converged,
        timings,
    )
