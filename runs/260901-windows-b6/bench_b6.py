r"""B6: BEAT CPU against bempp-OpenCL, on one box, one ladder, one set of frequencies.

Three engines, same meshes, same excitation:

  * ``bempp-fp64``  -- hornlab-bempp-bem, OpenCL backend, double precision
  * ``bempp-fp32``  -- hornlab-bempp-bem, OpenCL backend, single precision
  * ``beat-cpu``    -- hornlab-beat-bem with HORNLAB_BEAT_FORCE_CPU=1, defaults

Reading the numbers
-------------------

**Cold and warm are separated, and for BEAT that separation is the point.**
BEAT runs on Julia, so the first frequency in a fresh process pays Julia's JIT
exactly the way the first Numba call pays LLVM. Quoting a cold BEAT number
against a warm bempp number would be measuring the compiler. Every cell here
solves the same frequency twice in one process; ``cold`` is the first and
``warm`` the second, and the ratio table uses warm only.

**Neither engine reports singular and near-singular corrections as a separate
phase.** In bempp-cl they run inside the layer-potential assembly kernels. No
number is invented for them; ``assembly_s`` is whatever each engine attributes
to building the operator, and the phase dictionaries are copied verbatim into
the JSON so the split each engine *does* report stays inspectable.

**Agreement is scored on the far field, never on the surface, never on
absolute SPL, and never on null depth.** Nulls move by fractions of a degree
between discretizations and their depth is unbounded, so a null comparison
measures mesh luck. What is compared, per frequency:

  * main-lobe rms of the *normalized* pattern over 0-60 degrees, in dB, where
    each engine is normalized to its own on-axis level -- so this is pattern
    shape, not level;
  * the -6 dB beamwidth difference;
  * the on-axis complex-pressure ratio, as magnitude and phase.

**Impedance and delay sign are expected to disagree** and are recorded without
reconciliation: the two engines do not define the impedance output as the same
quantity, and a phase-convention difference shows up as a sign on delay.

    python bench_b6.py --julia <julia.exe> --out .
    python bench_b6.py --single L2 beat-cpu 2000 --julia <julia.exe>
"""
from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LADDER = HERE.parent / "260901-windows-b0" / "ladder"
ENGINES = ("bempp-fp64", "bempp-fp32", "beat-cpu")
RUNGS = ("L1", "L2", "L3")
SPOT_FREQUENCIES = (2000.0, 20000.0)
SWEEP_RUNG = "L2"
SWEEP_POINTS = 12
SWEEP_RANGE_HZ = (100.0, 20000.0)
ANGLE_COUNT = 37          # 5-degree steps over 0-180
MAIN_LOBE_MAX_DEG = 60.0
CELL_TIMEOUT_S = 1800
MESH_SCALE = 0.001   # the ladder is in millimetres


def _jsonable(obj):
    """json.dumps default= hook.

    Engine timing dictionaries are copied verbatim, and BEAT puts numpy arrays
    and numpy scalars inside them, which the stdlib encoder refuses. Converting
    here keeps the copy verbatim rather than pruning fields to whatever happens
    to serialize.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    if isinstance(obj, Path):
        return obj.name
    return str(obj)


# --------------------------------------------------------------------------
# measurement helpers
# --------------------------------------------------------------------------
def peak_rss_bytes() -> int | None:
    if sys.platform != "win32":
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        except Exception:
            return None

    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_uint32), ("PageFaultCount", ctypes.c_uint32),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]

    counters = PMC()
    counters.cb = ctypes.sizeof(counters)
    get_current = ctypes.windll.kernel32.GetCurrentProcess
    get_current.restype = ctypes.c_void_p
    get_info = ctypes.windll.psapi.GetProcessMemoryInfo
    get_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32]
    if not get_info(get_current(), ctypes.byref(counters), counters.cb):
        return None
    return int(counters.PeakWorkingSetSize)


class ProcessTreeAccounting:
    """CPU time and peak memory for this process *and its children*.

    Needed because BEAT runs the solve in a Julia child process. Measured from
    the parent, ``time.process_time()`` reports ~0.00 for a BEAT cell and
    ``PeakWorkingSetSize`` reports only the Python interpreter -- so the naive
    numbers would say BEAT used no CPU and 35 MB, which is nonsense. A Windows
    job object accounts for the whole tree, and children inherit it.

    Falls back to parent-only measurement when the job cannot be created, and
    says so through ``available`` rather than silently reporting a wrong number.
    The job is deliberately created *without* JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    so that closing it never tears the tree down.
    """

    def __init__(self) -> None:
        self.available = False
        self._job = None
        if sys.platform != "win32":
            return
        try:
            kernel32 = ctypes.windll.kernel32
            # argtypes are not optional here: with restype set to c_void_p the
            # GetCurrentProcess pseudo-handle is 0xFFFFFFFFFFFFFFFF, and ctypes'
            # default c_int conversion raises OverflowError on it -- which
            # presents as "job objects are unavailable" rather than as a bug.
            kernel32.CreateJobObjectW.restype = ctypes.c_void_p
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p,
                                                          ctypes.c_void_p]
            kernel32.AssignProcessToJobObject.restype = ctypes.c_int
            kernel32.QueryInformationJobObject.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                ctypes.c_uint32, ctypes.c_void_p]
            kernel32.QueryInformationJobObject.restype = ctypes.c_int

            job = kernel32.CreateJobObjectW(None, None)
            if not job:
                return
            if not kernel32.AssignProcessToJobObject(job,
                                                     kernel32.GetCurrentProcess()):
                return
            self._job = job
            self.available = True
        except Exception:
            self.available = False

    def _basic_accounting(self):
        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in
                        ("ReadOperationCount", "WriteOperationCount",
                         "OtherOperationCount", "ReadTransferCount",
                         "WriteTransferCount", "OtherTransferCount")]

        class BASIC_ACCOUNTING(ctypes.Structure):
            _fields_ = [
                ("TotalUserTime", ctypes.c_int64),
                ("TotalKernelTime", ctypes.c_int64),
                ("ThisPeriodTotalUserTime", ctypes.c_int64),
                ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                ("TotalPageFaultCount", ctypes.c_uint32),
                ("TotalProcesses", ctypes.c_uint32),
                ("ActiveProcesses", ctypes.c_uint32),
                ("TotalTerminatedProcesses", ctypes.c_uint32),
            ]

        info = BASIC_ACCOUNTING()
        ok = ctypes.windll.kernel32.QueryInformationJobObject(
            self._job, 1, ctypes.byref(info), ctypes.sizeof(info), None)
        return info if ok else None

    def cpu_seconds(self) -> float | None:
        """User+kernel CPU across the tree, in seconds (100 ns ticks)."""
        if not self.available:
            return None
        info = self._basic_accounting()
        if info is None:
            return None
        return (info.TotalUserTime + info.TotalKernelTime) / 1e7

    def peak_memory_bytes(self) -> int | None:
        if not self.available:
            return None

        class BASIC_LIMIT(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                        ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", ctypes.c_uint32),
                        ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t),
                        ("ActiveProcessLimit", ctypes.c_uint32),
                        ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                        ("PriorityClass", ctypes.c_uint32),
                        ("SchedulingClass", ctypes.c_uint32)]

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_uint64) for n in
                        ("ReadOperationCount", "WriteOperationCount",
                         "OtherOperationCount", "ReadTransferCount",
                         "WriteTransferCount", "OtherTransferCount")]

        class EXTENDED_LIMIT(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BASIC_LIMIT),
                        ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t),
                        ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t),
                        ("PeakJobMemoryUsed", ctypes.c_size_t)]

        info = EXTENDED_LIMIT()
        ok = ctypes.windll.kernel32.QueryInformationJobObject(
            self._job, 9, ctypes.byref(info), ctypes.sizeof(info), None)
        return int(info.PeakJobMemoryUsed) if ok else None


ACCOUNTING = ProcessTreeAccounting()


def expected_driver_area_m2(mesh_path: Path, driver_tag: int = 2) -> float:
    """Area of the driver patch in m^2, read straight from the mesh file.

    The reference value for the geometry check below. Both engines are asked
    what geometry they think they got, and a scale mistake shows up here as a
    factor of a million rather than as a puzzling disagreement in the far field.
    """
    lines = mesh_path.read_text(encoding="ascii").splitlines()
    ni = lines.index("$Nodes")
    n = int(lines[ni + 1])
    vertices = np.array([[float(v) for v in lines[ni + 2 + i].split()[1:4]]
                         for i in range(n)]) * MESH_SCALE
    ei = lines.index("$Elements")
    m = int(lines[ei + 1])
    total = 0.0
    for i in range(m):
        parts = lines[ei + 2 + i].split()
        if int(parts[3]) != driver_tag:
            continue
        a, b, c = (vertices[int(x) - 1] for x in parts[5:8])
        total += 0.5 * float(np.linalg.norm(np.cross(b - a, c - a)))
    return total


def ladder_meshes() -> dict[str, Path]:
    manifest = json.loads((LADDER / "ladder.json").read_text(encoding="utf-8"))
    return {f"L{i + 1}": LADDER / r["path"] for i, r in enumerate(manifest["rungs"])}


def describe_environment(julia: str | None) -> dict:
    env: dict = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }
    for module in ("numpy", "scipy", "numba", "pyopencl", "bempp_cl"):
        try:
            env[module] = getattr(__import__(module), "__version__", "unknown")
        except Exception:
            env[module] = "unavailable"
    if julia:
        # The interpreter is recorded by version and file name only: an
        # absolute path here would be a local path in a public artefact.
        env["julia_executable_name"] = Path(julia).name
        try:
            out = subprocess.run([julia, "--version"], capture_output=True,
                                 text=True, timeout=120)
            env["julia_version"] = out.stdout.strip() or out.stderr.strip()
        except Exception as exc:
            env["julia_version"] = f"unavailable: {exc}"
    return env


# --------------------------------------------------------------------------
# engines
# --------------------------------------------------------------------------
def _bempp_cell(mesh_path: Path, frequency: float, precision: str) -> dict:
    import hornlab_bempp_bem as bempp_bem
    from hornlab_bempp_bem.config import LinearSolver
    from hornlab_bempp_bem.mesh import load_mesh

    # scale= belongs on load_mesh, not on SolveConfig. SolveConfig.mesh_scale
    # only takes effect when the solver loads the mesh itself: _resolve_mesh
    # returns an already-loaded LoadedMesh untouched, and a test pins that it
    # must not reload. Passing a pre-loaded mesh plus mesh_scale therefore
    # scales nothing and silently solves the horn in millimetres-as-metres --
    # a 411 m horn instead of a 411 mm one, which showed up as a -6 dB
    # beamwidth of 2.5 degrees at 2 kHz and a 90 dB on-axis disagreement
    # against BEAT (which takes a path and scales correctly).
    mesh = load_mesh(str(mesh_path), scale=MESH_SCALE)
    config = bempp_bem.SolveConfig(
        observation=bempp_bem.ObservationConfig(
            planes=["horizontal"], angle_min_deg=0.0, angle_max_deg=180.0,
            angle_count=ANGLE_COUNT, origin="throat",
        ),
        velocity_sources={2: 1.0},
        precision=precision,
        assembly_backend="opencl",
        # LU, not the default. Left at its default this package selects GMRES,
        # which on this exterior Neumann problem does not converge: every cell
        # of the first B6 run stopped at the 5000-iteration cap with
        # converged=False, at every rung and both frequencies. Those timings
        # are "5000 iterations", not "a solve", and the far field they produce
        # is not a solution -- it disagreed with BEAT by 80 dB on axis.
        # BEAT's default is a direct dense solve, so LU is also the
        # like-for-like choice. The non-converged run is kept alongside as
        # RESULTS-b6-bempp-default-gmres.json.
        solver=LinearSolver.LU,
    )
    freqs = np.array([frequency], dtype=np.float64)

    runs, pressure = [], None
    for stage in ("cold", "warm"):
        w0 = time.perf_counter()
        c0 = ACCOUNTING.cpu_seconds()
        result = bempp_bem.solve_frequencies(mesh, freqs, config)
        wall = time.perf_counter() - w0
        c1 = ACCOUNTING.cpu_seconds()
        cpu = None if (c0 is None or c1 is None) else c1 - c0
        entry = result.solver_log[0]
        pt = entry.get("phase_timings", {})
        runs.append({
            "stage": stage, "wall_s": wall, "cpu_s": cpu,
            "cpu_over_wall": (cpu / wall) if (cpu is not None and wall) else None,
            "cpu_measured_across_process_tree": ACCOUNTING.available,
            "assembly_s": (pt.get("operator_construction_s", 0.0)
                           + pt.get("slp_assembly_s", 0.0)
                           + pt.get("dlp_assembly_s", 0.0)
                           + pt.get("adlp_assembly_s", 0.0)
                           + pt.get("hyp_assembly_s", 0.0)),
            "singular_near_correction_s": None,
            "materialization_s": pt.get("lhs_materialization_s")
            or pt.get("operator_materialization_s"),
            "linear_solve_s": pt.get("linear_solve_s"),
            "field_evaluation_s": result.timings.get("directivity_s"),
            "iterations": entry.get("iterations"),
            "converged": entry.get("converged"),
            "effective_precision": entry.get("effective_precision"),
            "effective_solver": entry.get("effective_solver"),
            "effective_backend": entry.get("effective_backend"),
            "phase_timings": pt,
        })
        pressure = result.pressure_complex
        angles = result.observation_angles_deg
        impedance = result.impedance

    vertices = np.asarray(mesh.grid.vertices).T
    elements = np.asarray(mesh.grid.elements).T
    tags = np.asarray(mesh.physical_tags)
    tri = vertices[elements[tags == 2]]
    driver_area = float(0.5 * np.linalg.norm(
        np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum())

    return {
        "runs": runs,
        "angles_deg": np.asarray(angles).ravel().tolist(),
        "pressure_real": np.asarray(pressure).ravel().real.tolist(),
        "pressure_imag": np.asarray(pressure).ravel().imag.tolist(),
        "impedance_real": np.asarray(impedance).ravel().real.tolist(),
        "impedance_imag": np.asarray(impedance).ravel().imag.tolist(),
        "driver_area_m2": driver_area,
        "engine_n_vertices": int(mesh.info.n_vertices),
        "formulation": str(getattr(config, "formulation", "default")),
    }


def _beat_cell(mesh_path: Path, frequency: float) -> dict:
    import hornlab_beat_bem as beat

    config = beat.SolveConfig(
        freq_min_hz=frequency, freq_max_hz=frequency, freq_count=1,
        observation=beat.ObservationConfig(
            planes=["horizontal"], angle_min_deg=0.0, angle_max_deg=180.0,
            angle_count=ANGLE_COUNT, origin="throat",
        ),
        velocity_sources={2: 1.0},
        # Note the asymmetry with the bempp arm, which is easy to get wrong in
        # exactly the way that produces a fake engine disagreement: BEAT is
        # handed a *path* and loads the mesh itself, so mesh_scale is what
        # applies the millimetre conversion and it is required here. The bempp
        # arm is handed an already-loaded mesh, where mesh_scale is ignored and
        # the conversion has to go on load_mesh(scale=...) instead. Dropping
        # either one silently solves a 411 m horn.
        mesh_scale=MESH_SCALE,
    )
    freqs = [frequency]

    runs, pressure, angles, impedance = [], None, None, None
    diagnostics = None
    for stage in ("cold", "warm"):
        w0 = time.perf_counter()
        c0 = ACCOUNTING.cpu_seconds()
        result = beat.solve_frequencies(str(mesh_path), freqs, config)
        wall = time.perf_counter() - w0
        c1 = ACCOUNTING.cpu_seconds()
        cpu = None if (c0 is None or c1 is None) else c1 - c0

        entry = (result.solver_log or [{}])[0]
        entry = entry if isinstance(entry, dict) else {}
        # BEAT nests its phase timings one level down, under "timings", and
        # names them assembly_s / solve_s / field_s.
        phases = entry.get("timings") or {}
        diagnostics = entry.get("native_diagnostics") or {}
        runs.append({
            "stage": stage, "wall_s": wall, "cpu_s": cpu,
            "cpu_over_wall": (cpu / wall) if (cpu is not None and wall) else None,
            "cpu_measured_across_process_tree": ACCOUNTING.available,
            "assembly_s": phases.get("assembly_s"),
            # BEAT reports no separate singular/near-correction phase either;
            # like bempp-cl it folds them into assembly.
            "singular_near_correction_s": None,
            "materialization_s": None,
            "linear_solve_s": phases.get("solve_s"),
            "field_evaluation_s": phases.get("field_s"),
            # Its direct dense solve is non-iterative, so there is no count.
            "iterations": entry.get("iterations"),
            "converged": entry.get("converged"),
            "effective_precision": diagnostics.get("precision", "float64"),
            "effective_solver": diagnostics.get("message"),
            "effective_backend": diagnostics.get("backend"),
            "blas_threads": diagnostics.get("blas_threads"),
            "phase_timings": phases,
            "native_diagnostics": diagnostics,
            "engine_timings": result.timings,
        })
        pressure = result.pressure_complex
        angles = result.observation_angles_deg
        impedance = result.impedance

    return {
        "runs": runs,
        "angles_deg": np.asarray(angles).ravel().tolist(),
        "pressure_real": np.asarray(pressure).ravel().real.tolist(),
        "pressure_imag": np.asarray(pressure).ravel().imag.tolist(),
        "impedance_real": np.asarray(impedance).ravel().real.tolist(),
        "impedance_imag": np.asarray(impedance).ravel().imag.tolist(),
        # Recorded rather than assumed: BEAT defaults to wavelength-adaptive
        # regular quadrature, which the bempp side on this base does not have.
        "driver_area_m2": (
            (getattr(result.mesh_info, "physical_tag_areas_m2", None) or {}).get(2)),
        "engine_n_vertices": getattr(result.mesh_info, "n_vertices", None),
        "formulation": {
            "regular_quadrature_mode": diagnostics.get("regular_quadrature_mode"),
            "regular_quadrature_order": diagnostics.get("regular_quadrature_order"),
            "regular_quadrature_base_order": diagnostics.get(
                "regular_quadrature_base_order"),
            "symmetry": diagnostics.get("symmetry"),
            "solver": diagnostics.get("message"),
        },
    }


def _entry_total_s(entry: dict, phases: dict) -> float | None:
    """Total seconds for one frequency, whichever way the engine reports it."""
    if entry.get("timing_s") is not None:
        return float(entry["timing_s"])
    if phases.get("total_s") is not None:
        return float(phases["total_s"])
    parts = [v for k, v in phases.items()
             if k.endswith("_s") and isinstance(v, (int, float))]
    return float(sum(parts)) if parts else None


def run_sweep(rung: str, engine: str, frequencies: list[float]) -> dict:
    """One process, all frequencies, in order -- the way a sweep actually runs.

    Doing this as N independent cold cells would pay Julia's JIT (or Numba's,
    or the OpenCL kernel build) once per point, which is not what a sweep costs
    and would bury the per-frequency signal under compilation. The first
    frequency here is still cold and is reported as such; the rest are warm.
    """
    mesh = ladder_meshes()[rung]
    freqs = np.asarray(frequencies, dtype=np.float64)

    wall0 = time.perf_counter()
    cpu0 = ACCOUNTING.cpu_seconds()
    if engine == "beat-cpu":
        import hornlab_beat_bem as beat

        config = beat.SolveConfig(
            freq_min_hz=float(freqs[0]), freq_max_hz=float(freqs[-1]),
            freq_count=len(freqs),
            observation=beat.ObservationConfig(
                planes=["horizontal"], angle_min_deg=0.0, angle_max_deg=180.0,
                angle_count=ANGLE_COUNT, origin="throat",
            ),
            velocity_sources={2: 1.0}, mesh_scale=MESH_SCALE,
        )
        result = beat.solve_frequencies(str(mesh), freqs.tolist(), config)
    else:
        import hornlab_bempp_bem as bempp_bem
        from hornlab_bempp_bem.mesh import load_mesh

        from hornlab_bempp_bem.config import LinearSolver

        config = bempp_bem.SolveConfig(
            observation=bempp_bem.ObservationConfig(
                planes=["horizontal"], angle_min_deg=0.0, angle_max_deg=180.0,
                angle_count=ANGLE_COUNT, origin="throat",
            ),
            velocity_sources={2: 1.0},
            precision="single" if engine.endswith("fp32") else "double",
            # See _bempp_cell: the default GMRES does not converge here.
            solver=LinearSolver.LU,
            assembly_backend="opencl",
        )
        result = bempp_bem.solve_frequencies(
            load_mesh(str(mesh), scale=MESH_SCALE), freqs, config)
    wall = time.perf_counter() - wall0
    cpu1 = ACCOUNTING.cpu_seconds()
    cpu = None if (cpu0 is None or cpu1 is None) else cpu1 - cpu0

    per_frequency = []
    for index, entry in enumerate(result.solver_log or []):
        entry = entry if isinstance(entry, dict) else {}
        # The two engines nest this differently: bempp puts a flat timing_s and
        # a phase_timings dict on the entry, BEAT puts everything under
        # "timings". Reading only bempp's shape silently yields a sweep of
        # None for BEAT.
        engine_phases = entry.get("phase_timings") or entry.get("timings") or {}
        per_frequency.append({
            "index": index,
            "stage": "cold" if index == 0 else "warm",
            "frequency_hz": entry.get("frequency_hz"),
            # bempp gives a flat timing_s; BEAT gives neither that nor a
            # per-frequency total_s, only the individual phases -- so fall all
            # the way through to summing them rather than recording None.
            "timing_s": _entry_total_s(entry, engine_phases),
            "iterations": entry.get("iterations"),
            "converged": entry.get("converged"),
            "effective_precision": entry.get("effective_precision"),
            "effective_solver": entry.get("effective_solver"),
            "effective_backend": entry.get("effective_backend"),
            "phase_timings": engine_phases,
        })

    pressure = np.asarray(result.pressure_complex)
    return {
        "rung": rung, "engine": engine, "mesh": mesh.name,
        "frequencies_hz": freqs.tolist(),
        "total_wall_s": wall, "total_cpu_s": cpu,
        "cpu_over_wall": (cpu / wall) if (cpu is not None and wall) else None,
        "engine_timings": result.timings,
        "per_frequency": per_frequency,
        "peak_memory_bytes": ACCOUNTING.peak_memory_bytes() or peak_rss_bytes(),
        "peak_memory_is_process_tree": ACCOUNTING.available,
        "angles_deg": np.asarray(result.observation_angles_deg).ravel().tolist(),
        "pressure_real": pressure.reshape(len(freqs), -1).real.tolist(),
        "pressure_imag": pressure.reshape(len(freqs), -1).imag.tolist(),
        "impedance_real": np.asarray(result.impedance).ravel().real.tolist(),
        "impedance_imag": np.asarray(result.impedance).ravel().imag.tolist(),
    }


def run_cell(rung: str, engine: str, frequency: float) -> dict:
    mesh = ladder_meshes()[rung]
    if engine == "beat-cpu":
        payload = _beat_cell(mesh, frequency)
    else:
        payload = _bempp_cell(
            mesh, frequency, "single" if engine.endswith("fp32") else "double")
    expected = expected_driver_area_m2(mesh)
    reported = payload.get("driver_area_m2")
    payload.update({
        "expected_driver_area_m2": expected,
        # A scale mistake is a factor of 1e6 here, so 5% is a generous gate.
        "geometry_check_passed": (
            None if reported is None else abs(reported - expected) <= 0.05 * expected),
        "rung": rung, "engine": engine, "frequency_hz": frequency,
        "mesh": mesh.name,
        # Tree-wide, because BEAT's solve lives in a Julia child process.
        "peak_memory_bytes": ACCOUNTING.peak_memory_bytes() or peak_rss_bytes(),
        "peak_memory_is_process_tree": ACCOUNTING.available,
    })
    return payload


# --------------------------------------------------------------------------
# agreement scoring
# --------------------------------------------------------------------------
def _pattern_db(pressure: np.ndarray) -> np.ndarray:
    magnitude = np.abs(pressure)
    reference = magnitude[0]
    if reference <= 0:
        return np.full_like(magnitude, np.nan, dtype=np.float64)
    return 20.0 * np.log10(np.maximum(magnitude / reference, 1e-30))


def _beamwidth_deg(angles: np.ndarray, pattern: np.ndarray, drop_db: float = -6.0):
    """First crossing of drop_db, linearly interpolated. None if it never drops."""
    for i in range(1, len(pattern)):
        if pattern[i] <= drop_db:
            a0, a1 = angles[i - 1], angles[i]
            p0, p1 = pattern[i - 1], pattern[i]
            if p1 == p0:
                return float(a1)
            return float(a0 + (drop_db - p0) * (a1 - a0) / (p1 - p0))
    return None


def score_agreement(reference: dict, candidate: dict) -> dict:
    angles = np.asarray(reference["angles_deg"], dtype=np.float64)
    ref = np.asarray(reference["pressure_real"]) + 1j * np.asarray(reference["pressure_imag"])
    cand = np.asarray(candidate["pressure_real"]) + 1j * np.asarray(candidate["pressure_imag"])
    if ref.shape != cand.shape:
        return {"error": f"shape mismatch {ref.shape} vs {cand.shape}"}

    ref_db, cand_db = _pattern_db(ref), _pattern_db(cand)
    main = angles <= MAIN_LOBE_MAX_DEG
    delta = cand_db[main] - ref_db[main]
    rms = float(np.sqrt(np.nanmean(delta ** 2)))

    bw_ref = _beamwidth_deg(angles, ref_db)
    bw_cand = _beamwidth_deg(angles, cand_db)

    ratio = cand[0] / ref[0] if ref[0] != 0 else complex("nan")
    return {
        "main_lobe_rms_db_0_60": rms,
        "main_lobe_max_abs_db_0_60": float(np.nanmax(np.abs(delta))),
        "beamwidth_6db_reference_deg": bw_ref,
        "beamwidth_6db_candidate_deg": bw_cand,
        "beamwidth_6db_delta_deg": (None if bw_ref is None or bw_cand is None
                                    else float(bw_cand - bw_ref)),
        "on_axis_magnitude_ratio": float(abs(ratio)),
        "on_axis_magnitude_ratio_db": float(20.0 * math.log10(abs(ratio)))
        if abs(ratio) > 0 else None,
        "on_axis_phase_difference_deg": float(np.degrees(np.angle(ratio))),
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _subprocess_env(julia: str | None) -> dict:
    env = os.environ.copy()
    env["HORNLAB_BEAT_FORCE_CPU"] = "1"
    if julia:
        env["HORNLAB_BEAT_JULIA"] = julia
    return env


def _invoke(rung: str, engine: str, frequency: float, julia: str | None) -> dict | None:
    proc = subprocess.run(
        [sys.executable, "-u", str(Path(__file__).resolve()),
         "--single", rung, engine, str(frequency)],
        capture_output=True, text=True, env=_subprocess_env(julia),
        timeout=CELL_TIMEOUT_S,
    )
    if proc.returncode != 0:
        print(f"  FAILED rc={proc.returncode}: {proc.stderr[-600:]}", flush=True)
        return {"rung": rung, "engine": engine, "frequency_hz": frequency,
                "error": proc.stderr[-2000:]}
    for line in reversed(proc.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    return {"rung": rung, "engine": engine, "frequency_hz": frequency,
            "error": "no JSON on stdout"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(HERE))
    parser.add_argument("--julia", help="Julia executable for the BEAT engine")
    parser.add_argument("--single", nargs=3, metavar=("RUNG", "ENGINE", "FREQ"))
    parser.add_argument("--sweep-engine", help="run the sweep for one engine (internal)")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--only-sweep", action="store_true")
    args = parser.parse_args(argv)

    if args.single:
        rung, engine, freq = args.single
        print(json.dumps(run_cell(rung, engine, float(freq)), default=_jsonable))
        return 0

    if args.sweep_engine:
        freqs = np.geomspace(*SWEEP_RANGE_HZ, SWEEP_POINTS).tolist()
        print(json.dumps(run_sweep(SWEEP_RUNG, args.sweep_engine, freqs), default=_jsonable))
        return 0

    julia = args.julia or os.environ.get("HORNLAB_BEAT_JULIA") or shutil.which("julia")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cells: list[dict] = []
    for rung in ([] if args.only_sweep else RUNGS):
        for frequency in SPOT_FREQUENCIES:
            for engine in ENGINES:
                print(f"[{rung} {engine:10s} {frequency:8.0f} Hz] running...", flush=True)
                cell = _invoke(rung, engine, frequency, julia)
                cells.append(cell)
                if "error" not in cell:
                    warm = cell["runs"][1]
                    print(f"  warm wall {warm['wall_s']:8.2f}s  "
                          f"cold wall {cell['runs'][0]['wall_s']:8.2f}s  "
                          f"cpu/wall {warm['cpu_over_wall'] or 0:5.2f}", flush=True)

    # Agreement, per (rung, frequency), against bempp-fp64.
    agreement = []
    for rung in RUNGS:
        for frequency in SPOT_FREQUENCIES:
            def find(engine):
                return next((c for c in cells
                             if c.get("rung") == rung and c.get("engine") == engine
                             and c.get("frequency_hz") == frequency
                             and "error" not in c), None)
            reference = find("bempp-fp64")
            if reference is None:
                continue
            for engine in ("bempp-fp32", "beat-cpu"):
                candidate = find(engine)
                if candidate is None:
                    continue
                agreement.append({
                    "rung": rung, "frequency_hz": frequency,
                    "reference": "bempp-fp64", "candidate": engine,
                    **score_agreement(reference, candidate),
                })

    bundle = {
        "benchmark": "B6 BEAT-CPU vs bempp-OpenCL",
        "environment": describe_environment(julia),
        "ladder": json.loads((LADDER / "ladder.json").read_text(encoding="utf-8")),
        "note_coupled": (
            "BEAT's coupled CPU solver is not reachable through the Python "
            "wrapper: SolveConfig exposes no coupled/chamber/interior field."
        ),
        "cells": cells,
        "agreement": agreement,
    }
    if not args.only_sweep:
        (out_dir / "RESULTS-b6.json").write_text(
            json.dumps(bundle, indent=2, default=_jsonable), encoding="utf-8")
        print("wrote RESULTS-b6.json", flush=True)

    if not args.skip_sweep:
        frequencies = np.geomspace(*SWEEP_RANGE_HZ, SWEEP_POINTS)
        sweep = []
        for engine in ENGINES:
            print(f"[sweep {engine:10s} {SWEEP_POINTS} points in one process]", flush=True)
            proc = subprocess.run(
                [sys.executable, "-u", str(Path(__file__).resolve()),
                 "--sweep-engine", engine],
                capture_output=True, text=True, env=_subprocess_env(julia),
                timeout=CELL_TIMEOUT_S * 3,
            )
            if proc.returncode != 0:
                print(f"  FAILED rc={proc.returncode}: {proc.stderr[-600:]}", flush=True)
                sweep.append({"engine": engine, "error": proc.stderr[-2000:]})
                continue
            line = next((x for x in reversed(proc.stdout.strip().splitlines())
                         if x.startswith("{")), None)
            record = json.loads(line) if line else {"engine": engine,
                                                   "error": "no JSON on stdout"}
            sweep.append(record)
            if "error" not in record:
                print(f"  total wall {record['total_wall_s']:8.2f}s  "
                      f"cpu/wall {record['cpu_over_wall'] or 0:5.2f}", flush=True)
        (out_dir / "sweep_L2.json").write_text(json.dumps({
            "benchmark": "B6 12-point log sweep",
            "rung": SWEEP_RUNG,
            "frequencies_hz": frequencies.tolist(),
            "environment": describe_environment(julia),
            "cells": sweep,
        }, indent=2, default=_jsonable), encoding="utf-8")
        print("wrote sweep_L2.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
