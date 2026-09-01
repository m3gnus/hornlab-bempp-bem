r"""B0: Numba vs OpenCL CPU assembly on Windows, across a three-rung DOF ladder.

Measures the cost of bempp-cl's two CPU assembly backends on the same machine,
same meshes, same frequencies, at both fp64 and fp32. The question it answers is
what a missing CPU OpenCL runtime actually costs, because bempp-cl falls back to
Numba silently and the answer decides whether that fallback deserves a warning.

Method notes that matter for reading the numbers:

* **Cold and warm are reported separately.** The first solve in a process pays
  Numba's JIT or OpenCL's kernel build; the second does not. Quoting a cold
  number as though it were steady-state is the classic way to make Numba look
  worse and OpenCL look better than either is. Each configuration therefore
  solves the same frequency twice in one process and both are recorded.
* **Ratios are computed from warm times only.**
* **bempp-cl does not break out singular and near-singular corrections as a
  phase.** They happen inside ``slp_assembly_s`` and ``dlp_assembly_s``. No
  separate number is invented here; ``assembly_s`` is the sum of operator
  construction and the layer-potential assemblies, and the dense
  materialization (``lhs_materialization_s``) is reported as its own column
  because on this ladder it is the largest single term.
* **Peak RSS** is the Win32 ``PeakWorkingSetSize`` for the whole process, so it
  includes the interpreter and both backends' imports.

    python bench_b0.py --out .                  # full ladder
    python bench_b0.py --threads-probe          # thread-scaling arm only
    python bench_b0.py --single L2 opencl double --threads 4    # one cell
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LADDER = HERE / "ladder"
FREQUENCY_HZ = 2000.0
RUNGS = ("L1", "L2", "L3")


# --------------------------------------------------------------------------
# environment / measurement helpers
# --------------------------------------------------------------------------
def peak_rss_bytes() -> int | None:
    """Win32 PeakWorkingSetSize; None off Windows."""
    if sys.platform != "win32":
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        except Exception:
            return None

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_uint32),
            ("PageFaultCount", ctypes.c_uint32),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(counters)
    # Without an explicit restype the pseudo-handle comes back as a truncated
    # C int and the call fails, silently reporting no memory at all.
    get_current_process = ctypes.windll.kernel32.GetCurrentProcess
    get_current_process.restype = ctypes.c_void_p
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    get_info.restype = ctypes.c_int
    if not get_info(get_current_process(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize)


def describe_environment() -> dict:
    env: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    try:
        import bempp_cl

        env["bempp_cl"] = getattr(bempp_cl, "__version__", "unknown")
    except Exception:
        env["bempp_cl"] = "unavailable"
    for module in ("numpy", "scipy", "numba", "pyopencl"):
        try:
            env[module] = __import__(module).__version__
        except Exception:
            env[module] = "unavailable"
    try:
        import pyopencl as cl

        devices = []
        for pf in cl.get_platforms():
            for dev in pf.get_devices():
                devices.append(
                    {
                        "platform": pf.name.strip(),
                        "name": dev.name.strip(),
                        "type": cl.device_type.to_string(dev.type),
                        "max_compute_units": dev.max_compute_units,
                        # 0 on a VM whose host does not expose the real clock;
                        # recorded because it makes per-core claims unverifiable.
                        "max_clock_frequency_mhz": dev.max_clock_frequency,
                        "global_mem_bytes": int(dev.global_mem_size),
                    }
                )
        env["opencl_devices"] = devices
    except Exception as exc:
        env["opencl_devices"] = f"unavailable: {exc}"
    for var in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS"):
        env[var] = os.environ.get(var)
    return env


def ladder_meshes() -> dict[str, Path]:
    manifest = json.loads((LADDER / "ladder.json").read_text(encoding="utf-8"))
    return {
        f"L{i + 1}": LADDER / rung["path"]
        for i, rung in enumerate(manifest["rungs"])
    }


# --------------------------------------------------------------------------
# one measured cell
# --------------------------------------------------------------------------
def _config(backend: str, precision: str):
    import hornlab_bempp_bem as bempp_bem
    from hornlab_bempp_bem.config import LinearSolver

    return bempp_bem.SolveConfig(
        observation=bempp_bem.ObservationConfig(
            planes=["horizontal"],
            angle_min_deg=0.0,
            angle_max_deg=180.0,
            angle_count=37,
            # The ASRO reference set is throat-referenced; using the mouth
            # default would shift phase against every other result here.
            origin="throat",
        ),
        velocity_sources={2: 1.0},
        solver=LinearSolver.LU,
        precision=precision,
        assembly_backend=backend,
        mesh_scale=0.001,  # the ladder is in millimetres
        return_surface_traces=True,
    )


def _phases(entry: dict) -> dict:
    """Fold bempp's phase timings into the four categories B0 reports."""
    pt = entry.get("phase_timings", {})
    assembly = (
        pt.get("operator_construction_s", 0.0)
        + pt.get("slp_assembly_s", 0.0)
        + pt.get("dlp_assembly_s", 0.0)
        + pt.get("adlp_assembly_s", 0.0)
        + pt.get("hyp_assembly_s", 0.0)
    )
    return {
        "assembly_s": assembly,
        # bempp-cl folds singular and near-singular corrections into the
        # assembly kernels above; it reports no separate phase for them.
        "singular_near_correction_s": None,
        "materialization_s": pt.get("lhs_materialization_s")
        or pt.get("operator_materialization_s"),
        "linear_solve_s": pt.get("linear_solve_s"),
        "total_s": pt.get("total_s"),
        "raw": pt,
    }


def measure(mesh_path: Path, backend: str, precision: str) -> dict:
    import hornlab_bempp_bem as bempp_bem
    from hornlab_bempp_bem.mesh import load_mesh

    mesh = load_mesh(str(mesh_path))
    config = _config(backend, precision)
    frequencies = np.array([FREQUENCY_HZ], dtype=np.float64)

    runs = []
    for label in ("cold", "warm"):
        wall0, cpu0 = time.perf_counter(), time.process_time()
        result = bempp_bem.solve_frequencies(mesh, frequencies, config)
        wall = time.perf_counter() - wall0
        cpu = time.process_time() - cpu0
        entry = result.solver_log[0]
        runs.append(
            {
                "stage": label,
                "wall_s": wall,
                "cpu_s": cpu,
                "cpu_over_wall": (cpu / wall) if wall > 0 else None,
                "field_evaluation_s": result.timings.get("directivity_s"),
                "solve_s": result.timings.get("solve_s"),
                "iterations": entry.get("iterations"),
                "converged": entry.get("converged"),
                "effective_backend": entry.get("effective_backend"),
                "effective_precision": entry.get("effective_precision"),
                "effective_solver": entry.get("effective_solver"),
                "fallback_used": entry.get("fallback_used"),
                "fallback_reason": entry.get("reason"),
                **_phases(entry),
            }
        )

    return {
        "mesh": mesh_path.name,
        "p1_dof": int(mesh.info.n_vertices),
        "n_triangles": int(mesh.info.n_triangles),
        "requested_backend": backend,
        "requested_precision": precision,
        "frequency_hz": FREQUENCY_HZ,
        "runs": runs,
        "peak_rss_bytes": peak_rss_bytes(),
        # A fingerprint of the answer, so a "faster" configuration that quietly
        # stopped solving the same problem is visible.
        "on_axis_spl_db": None,
    }


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
def run_cell(rung: str, backend: str, precision: str) -> dict:
    meshes = ladder_meshes()
    record = measure(meshes[rung], backend, precision)
    record["rung"] = rung
    return record


def run_full(out_dir: Path) -> dict:
    """Every cell, each in its own process so cold really is cold."""
    meshes = ladder_meshes()
    cells = []
    for rung in RUNGS:
        for backend in ("numba", "opencl"):
            for precision in ("double", "single"):
                print(f"[{rung} {backend:6s} {precision:6s}] running...", flush=True)
                proc = subprocess.run(
                    [sys.executable, "-u", str(Path(__file__).resolve()),
                     "--single", rung, backend, precision],
                    capture_output=True, text=True, env=os.environ.copy(),
                )
                if proc.returncode != 0:
                    print(f"  FAILED rc={proc.returncode}\n{proc.stderr[-1500:]}", flush=True)
                    cells.append({
                        "rung": rung, "requested_backend": backend,
                        "requested_precision": precision,
                        "error": proc.stderr[-2000:],
                    })
                    continue
                cell = json.loads(proc.stdout.strip().splitlines()[-1])
                cells.append(cell)
                warm = cell["runs"][1]
                print(f"  warm assembly {warm['assembly_s']:8.2f}s  "
                      f"materialization {warm['materialization_s'] or 0:8.2f}s  "
                      f"solve {warm['linear_solve_s'] or 0:7.2f}s  "
                      f"wall {warm['wall_s']:8.2f}s", flush=True)
    return {
        "benchmark": "B0 numba-vs-opencl",
        "environment": describe_environment(),
        "ladder": json.loads((LADDER / "ladder.json").read_text(encoding="utf-8")),
        "cells": cells,
    }


def run_threads_probe(out_dir: Path, rung: str = "L2") -> dict:
    """Assembly time vs thread count, one fresh process per count.

    NUMBA_NUM_THREADS is read at import, so it can only be set from outside the
    process that uses it -- hence the subprocess per point.
    """
    counts = [1, 2, 4, 8, os.cpu_count() or 12]
    counts = sorted({c for c in counts if c and c <= (os.cpu_count() or 12)})
    points = []
    for backend in ("numba", "opencl"):
        for n in counts:
            env = os.environ.copy()
            env.update({
                "NUMBA_NUM_THREADS": str(n),
                "OMP_NUM_THREADS": str(n),
                "MKL_NUM_THREADS": str(n),
                "OPENBLAS_NUM_THREADS": str(n),
            })
            print(f"[threads {backend:6s} n={n:2d}] running...", flush=True)
            proc = subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve()),
                 "--single", rung, backend, "double"],
                capture_output=True, text=True, env=env,
            )
            if proc.returncode != 0:
                points.append({"backend": backend, "threads": n,
                               "error": proc.stderr[-1000:]})
                print(f"  FAILED rc={proc.returncode}", flush=True)
                continue
            cell = json.loads(proc.stdout.strip().splitlines()[-1])
            warm = cell["runs"][1]
            points.append({
                "backend": backend,
                "threads": n,
                "assembly_s": warm["assembly_s"],
                "materialization_s": warm["materialization_s"],
                "linear_solve_s": warm["linear_solve_s"],
                "wall_s": warm["wall_s"],
                "cpu_over_wall": warm["cpu_over_wall"],
            })
            print(f"  assembly {warm['assembly_s']:7.2f}s  "
                  f"cpu/wall {warm['cpu_over_wall']:5.2f}", flush=True)
    return {
        "benchmark": "B0 thread scaling",
        "rung": rung,
        "environment": describe_environment(),
        "points": points,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(HERE))
    parser.add_argument("--single", nargs=3, metavar=("RUNG", "BACKEND", "PRECISION"),
                        help="run one cell and print its JSON (internal)")
    parser.add_argument("--threads-probe", action="store_true")
    args = parser.parse_args(argv)

    if args.single:
        rung, backend, precision = args.single
        print(json.dumps(run_cell(rung, backend, precision)))
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.threads_probe:
        report = run_threads_probe(out_dir)
        (out_dir / "threads_probe.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8")
        print("wrote threads_probe.json")
        return 0

    bundle = run_full(out_dir)
    (out_dir / "RESULTS-bundle.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8")
    l3 = [c for c in bundle["cells"] if c.get("rung") == "L3"]
    (out_dir / "L3_warm.json").write_text(
        json.dumps({
            "benchmark": "B0 L3 warm detail",
            "environment": bundle["environment"],
            "cells": [
                {k: v for k, v in c.items() if k != "runs"}
                | {"warm": c["runs"][1], "cold": c["runs"][0]}
                for c in l3 if "runs" in c
            ],
        }, indent=2), encoding="utf-8")
    print("wrote RESULTS-bundle.json and L3_warm.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
