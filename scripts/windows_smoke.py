"""Prove a real solve works on this machine, and that OpenCL agrees with Numba.

Upstream bempp-cl runs no Windows or macOS CI, so a machine that can install
this package is not thereby known to be able to solve on it. This script is the
missing check. It is not a unit test -- it exercises the whole path a user
takes: import, device selection, kernel build, assembly, LU solve, and the
exterior field evaluation afterwards.

Two failures it is specifically built to catch:

* A missing OpenCL CPU runtime. bempp-cl falls back to Numba and says so only
  at info level, and the fallback is not cheap -- Numba assembly is several
  times the OpenCL time and the gap widens with mesh size (see the measured
  table in the README). Unannounced, that reads as a slow machine.
* An OpenCL runtime that is present and enumerable but wrong. Device
  enumeration and ``DEFAULT_DEVICE_INTERFACE`` both look healthy in that case,
  so the only way to catch it is to assemble on both backends and compare.

Exit status is 0 when everything checked passed, 1 otherwise, so it can be
wired into CI directly.

    python scripts/windows_smoke.py
    python scripts/windows_smoke.py --refinement 2 --json smoke.json
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

# Runnable straight from a checkout: Python puts scripts/ on sys.path, not the
# repo root, so an uninstalled tree would not import. An installed copy still
# wins, because this only appends.
sys.path.append(str(Path(__file__).resolve().parent.parent))


def _sphere_mesh(refinement: int):
    """A closed sphere with one element driven and the rest rigid."""
    import bempp_cl.api as bempp_api

    from hornlab_bempp_bem.mesh import LoadedMesh
    from hornlab_bempp_bem.result import MeshInfo

    grid = bempp_api.shapes.regular_sphere(refinement)
    vertices = np.asarray(grid.vertices).T
    n_elements = int(grid.number_of_elements)

    tags = np.ones(n_elements, dtype=np.int32)
    tags[0] = 2  # one driven element is enough to excite the problem

    return LoadedMesh(
        grid=grid,
        physical_tags=tags,
        info=MeshInfo(
            n_vertices=int(grid.number_of_vertices),
            n_triangles=n_elements,
            physical_groups={1: "rigid", 2: "source"},
            bounding_box_m=(vertices.min(axis=0), vertices.max(axis=0)),
        ),
    )


def _config(backend: str):
    import hornlab_bempp_bem as bempp_bem
    from hornlab_bempp_bem.config import LinearSolver

    points = np.array(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=np.float64
    )
    return bempp_bem.SolveConfig(
        observation=bempp_bem.ObservationConfig(
            planes=["probe"],
            angle_count=points.shape[0],
            custom_points={"probe": points},
        ),
        velocity_sources={2: 1.0},
        solver=LinearSolver.LU,
        precision="double",
        assembly_backend=backend,
        return_surface_traces=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--refinement", type=int, default=3,
        help="regular_sphere refinement; 3 is 512 elements (default: 3)",
    )
    parser.add_argument(
        "--rtol", type=float, default=1e-8,
        help="relative tolerance for OpenCL-vs-Numba agreement (default: 1e-8)",
    )
    parser.add_argument("--json", help="write the report to this path")
    args = parser.parse_args(argv)

    report: dict = {"checks": []}
    failures: list[str] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        report["checks"].append({"name": name, "ok": ok, "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""),
              flush=True)
        if not ok:
            failures.append(name)

    print("=" * 78)
    print("hornlab-bempp-bem smoke test")
    print("=" * 78)

    import bempp_cl
    import bempp_cl.api as bempp_api

    import hornlab_bempp_bem as bempp_bem

    report["environment"] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "bempp_cl": getattr(bempp_cl, "__version__", "unknown"),
        "numpy": np.__version__,
        "bempp_default_device_interface": str(bempp_api.DEFAULT_DEVICE_INTERFACE),
        "bempp_cpu_opencl_driver_found": bool(
            getattr(bempp_api, "CPU_OPENCL_DRIVER_FOUND", False)
        ),
    }
    for key, value in report["environment"].items():
        print(f"  {key:32s} {value}")
    print()

    print("OpenCL")
    check = bempp_bem.check_opencl("cpu")
    report["opencl_check"] = {
        "ok": check.ok, "stage": check.stage,
        "device_name": check.device_name, "detail": check.detail,
    }
    record("opencl device usable", check.ok, check.describe())
    print()

    mesh = _sphere_mesh(args.refinement)
    frequencies = np.array([1000.0, 5000.0], dtype=np.float64)
    report["mesh"] = {
        "n_vertices": mesh.info.n_vertices,
        "n_triangles": mesh.info.n_triangles,
        "refinement": args.refinement,
    }
    print(f"Mesh: {mesh.info.n_triangles} triangles, "
          f"{mesh.info.n_vertices} vertices; {frequencies.size} frequencies")
    print()

    backends = ["numba"] + (["opencl"] if check.ok else [])
    if not check.ok:
        print("  (skipping the OpenCL arm: no usable device)")
    results: dict[str, object] = {}
    report["solves"] = {}

    print("Solves")
    for backend in backends:
        try:
            t0 = time.perf_counter()
            solved = bempp_bem.solve_frequencies(mesh, frequencies, _config(backend))
            elapsed = time.perf_counter() - t0
            results[backend] = solved
            report["solves"][backend] = {"ok": True, "seconds": elapsed}
            record(f"solve on {backend}", True, f"{elapsed:.2f}s")
        except Exception as exc:
            report["solves"][backend] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            record(f"solve on {backend}", False, f"{type(exc).__name__}: {exc}")
    print()

    if len(results) == 2:
        print("Agreement")
        a = np.asarray(results["numba"].surface_pressure_complex)
        b = np.asarray(results["opencl"].surface_pressure_complex)
        denom = np.linalg.norm(a)
        rel = float(np.linalg.norm(a - b) / denom) if denom else float(np.linalg.norm(a - b))
        report["opencl_vs_numba_relative_difference"] = rel
        record(
            "opencl agrees with numba", rel <= args.rtol,
            f"relative difference {rel:.3e} (tolerance {args.rtol:.0e})",
        )
        if report["solves"]["opencl"]["ok"] and report["solves"]["numba"]["ok"]:
            speedup = report["solves"]["numba"]["seconds"] / report["solves"]["opencl"]["seconds"]
            report["opencl_speedup_over_numba"] = speedup
            # Reported, never asserted: this mesh is far too small for the
            # ratio to mean anything, and it is dominated by kernel build.
            print(f"  (opencl was {speedup:.2f}x numba here -- indicative only, "
                  f"this mesh is too small to benchmark)")
        print()

    report["ok"] = not failures
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"wrote {args.json}")

    print("=" * 78)
    if failures:
        print(f"SMOKE TEST FAILED: {len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
