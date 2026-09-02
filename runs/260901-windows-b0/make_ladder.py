r"""Build a three-rung P1-DOF ladder by remeshing the ASRO68 reference horn.

The ladder rungs are specified by *P1 degrees of freedom*, not by element size:
1,364 / 2,884 / 4,984. P1 DOF is the number of mesh vertices the solver actually
sees, which is not the node count in the .msh file -- ATH's export carries nodes
that belong to no triangle (197 of them in the reference mesh), and the loader
drops those. So the only honest way to hit a DOF target is to mesh, load through
the package, and count. That is what the bisection below does.

Geometry source is ATH's own ``bem_mesh.geo``, not the exported ``.msh``.
Remeshing the exported triangulation was tried first and does not work: gmsh
reparametrizes the discrete surface happily (``classifySurfaces`` in 0.2 s) but
then hangs in ``generate(2)``, emitting degenerate curve parametrizations with
denormal coordinates (``-2.122e-314``). The ``.geo`` is the real geometry and
meshes in under two seconds, so the ladder is built from it.

``Mesh.MeshSizeFactor`` must be set *before* ``gmsh.open()``. Setting it after
silently applies to the next mesh instead of the current one, which shows up as
a non-monotone and quietly wrong ladder.

The .geo defines no physical groups, so the driver patch is re-tagged
geometrically: the throat cap at the origin. The reference mesh puts its 64
driver elements at centroid radius 2.87-11.48 mm and z 0.25-1.57 mm, inside a
throat of radius 12.7 mm at z = 0, so "centroid radius < 12.7 mm and z < 2 mm"
reproduces that patch on any refinement.

    python make_ladder.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# The ASRO68 geometry is an external reference fixture that is not in this
# repository, so it is named the same way the test suite names its siblings:
# through an environment variable, not a path baked into the file.
GEO_ENV_VAR = "HORNLAB_ASRO68_GEO"

TARGET_DOF = (1364, 2884, 4984)


def geometry_path(explicit: str | None = None) -> Path:
    candidate = explicit or os.environ.get(GEO_ENV_VAR)
    if not candidate:
        raise SystemExit(
            f"Set {GEO_ENV_VAR} to ATH's bem_mesh.geo for the ASRO68 "
            "free-standing project, or pass --geo."
        )
    return Path(candidate)


RIGID_TAG, DRIVER_TAG = 1, 2
DRIVER_MAX_RADIUS_MM = 12.7
DRIVER_MAX_Z_MM = 2.0


def _mesh_at(factor: float, geo: Path):
    """Mesh the .geo at a mesh-size factor; return (vertices, triangles).

    A fresh gmsh session per call, because Mesh.MeshSizeFactor has to be set
    before the geometry is opened and reusing a session makes that easy to get
    wrong.
    """
    import gmsh

    # gmsh.initialize() truncates the Windows PATH (1486 -> 316 chars here),
    # which breaks any subprocess launched later in the same process.
    saved_path = os.environ.get("PATH", "")
    gmsh.initialize()
    try:
        os.environ["PATH"] = saved_path
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.option.setNumber("Mesh.MeshSizeFactor", factor)
        gmsh.open(str(geo))
        gmsh.model.mesh.generate(2)

        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        index = {int(t): i for i, t in enumerate(node_tags)}

        etypes, _, enodes = gmsh.model.mesh.getElements(2)
        tris = []
        for etype, conn in zip(etypes, enodes):
            if etype != 2:  # 3-node triangle
                continue
            conn = np.asarray(conn, dtype=np.int64).reshape(-1, 3)
            tris.append(np.vectorize(index.__getitem__)(conn))
        triangles = np.vstack(tris) if tris else np.zeros((0, 3), dtype=np.int64)
    finally:
        gmsh.finalize()
        os.environ["PATH"] = saved_path

    # Keep only vertices a triangle actually references, so the count written
    # here is the count the solver will report.
    used = np.unique(triangles)
    remap = np.full(coords.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    return _merge_coincident(coords[used], remap[triangles])


def _merge_coincident(vertices, triangles, decimals: int = 9):
    """Weld vertices that share a position, and renumber the triangles.

    gmsh emits one node per (surface, position) pair, so every seam between the
    20 surfaces its classification produces carries duplicated nodes -- 83 of
    them at the smallest rung here, 203 at the largest. That is not cosmetic:
    the two engines disagree about what to do with it. hornlab-bempp-bem welds
    them on load (``merge_tol=1e-9``) and reports 1,338 vertices; BEAT does not
    and reports 1,421 for the same file. So without this the two engines solve
    *different discretizations*, and BEAT's is a cracked surface whose P1 space
    is not continuous across the seams -- which is exactly the kind of thing
    that looks like an engine disagreement in a comparison table.

    Welding here means the file itself is watertight and every consumer sees
    the same mesh, rather than each applying its own policy.
    """
    keys = np.round(vertices, decimals)
    _, first_index, inverse = np.unique(
        keys, axis=0, return_index=True, return_inverse=True)
    # np.unique sorts; keep the original vertex order so the mesh stays stable.
    order = np.argsort(first_index)
    position = np.empty(order.size, dtype=np.int64)
    position[order] = np.arange(order.size)
    return vertices[first_index[order]], position[inverse.ravel()][triangles]


def _tag(vertices, triangles):
    centroids = vertices[triangles].mean(axis=1)
    radius = np.hypot(centroids[:, 0], centroids[:, 1])
    is_driver = (radius < DRIVER_MAX_RADIUS_MM) & (centroids[:, 2] < DRIVER_MAX_Z_MM)
    tags = np.full(triangles.shape[0], RIGID_TAG, dtype=np.int32)
    tags[is_driver] = DRIVER_TAG
    return tags


def _write_msh22(path: Path, vertices, triangles, tags) -> None:
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        fh.write('$PhysicalNames\n2\n2 1 "SD1G0"\n2 2 "SD1D1001"\n$EndPhysicalNames\n')
        fh.write(f"$Nodes\n{vertices.shape[0]}\n")
        for i, (x, y, z) in enumerate(vertices, start=1):
            fh.write(f"{i} {x:.10g} {y:.10g} {z:.10g}\n")
        fh.write("$EndNodes\n")
        fh.write(f"$Elements\n{triangles.shape[0]}\n")
        for i, (tri, tag) in enumerate(zip(triangles, tags), start=1):
            a, b, c = (int(v) + 1 for v in tri)
            fh.write(f"{i} 2 2 {int(tag)} {int(tag)} {a} {b} {c}\n")
        fh.write("$EndElements\n")


def _dof_of(path: Path) -> int:
    """P1 DOF as the solver counts it."""
    from hornlab_bempp_bem.mesh import load_mesh

    return int(load_mesh(str(path)).info.n_vertices)


def build(target: int, out_dir: Path, geo: Path, tol: int = 40, max_iter: int = 30):
    """Bisect Mesh.MeshSizeFactor until the loaded P1 DOF lands within tol."""
    lo, hi = 0.30, 8.0  # smaller factor = finer mesh = more DOF
    best = None
    best_path = None
    tmp = out_dir / f".probe_{target}.msh"

    for it in range(max_iter):
        mid = 0.5 * (lo + hi)
        vertices, triangles = _mesh_at(mid, geo)
        tags = _tag(vertices, triangles)
        _write_msh22(tmp, vertices, triangles, tags)
        dof = _dof_of(tmp)
        drivers = int((tags == DRIVER_TAG).sum())
        print(f"  target {target:5d}  iter {it:2d}  factor {mid:6.3f} -> dof {dof:6d} "
              f"tris {triangles.shape[0]:6d} driver {drivers:4d}", flush=True)

        if best is None or abs(dof - target) < abs(best[1] - target):
            best = (mid, dof, triangles.shape[0], drivers)
            final = out_dir / f"asro68_L{TARGET_DOF.index(target) + 1}_dof{dof}.msh"
            _write_msh22(final, vertices, triangles, tags)
            # Each improvement supersedes the last; without this the bisection
            # leaves every intermediate rung on disk.
            if best_path is not None and best_path != final:
                best_path.unlink(missing_ok=True)
            best_path = final

        if abs(dof - target) <= tol:
            break
        if dof > target:
            lo = mid          # too fine -> coarsen (larger factor)
        else:
            hi = mid
    tmp.unlink(missing_ok=True)

    factor, dof, ntri, drivers = best
    return {
        "target_dof": target,
        "mesh_size_factor": factor,
        "p1_dof": dof,
        "n_triangles": ntri,
        "n_driver_elements": drivers,
        "path": best_path.name,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(Path(__file__).parent / "ladder"))
    parser.add_argument("--geo", help=f"ATH bem_mesh.geo (default: ${GEO_ENV_VAR})")
    args = parser.parse_args(argv)

    geo = geometry_path(args.geo)
    if not geo.exists():
        print(f"ASRO68 geometry not found: {geo}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rungs = [build(t, out_dir, geo) for t in TARGET_DOF]
    manifest = {
        "geometry": "ASRO68 (ATH bem_mesh.geo, free-standing)",
        "units": "mm",
        "driver_patch": {
            "max_radius_mm": DRIVER_MAX_RADIUS_MM,
            "max_z_mm": DRIVER_MAX_Z_MM,
        },
        "rungs": rungs,
    }
    (out_dir / "ladder.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
