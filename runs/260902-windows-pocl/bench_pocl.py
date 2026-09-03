r"""B0's assembly ladder against PoCL, on the same meshes as the Intel run.

This reuses ``runs/260901-windows-b0/bench_b0.py`` unchanged -- same ladder,
same frequency, same process-per-cell shape, same warm/cold split -- and drives
only its OpenCL arm, selecting the runtime with ``BEMPP_CPU_DRIVER``. The Numba
and Intel columns are not re-run; they are read from the earlier bundle.

Read the fingerprint columns before the timing columns. On a runtime whose
kernel does not build, the timings are real measurements of doing nothing.

    set BEMPP_CPU_DRIVER=Portable
    python bench_pocl.py --out .
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
B0 = HERE.parent / "260901-windows-b0"
BENCH = B0 / "bench_b0.py"
RUNGS = ("L1", "L2", "L3")
PRECISIONS = ("double", "single")


def background_load(seconds: float = 3.0) -> dict:
    """How busy the machine is, sampled now rather than asserted afterwards.

    A stray worker burning a core inflated an unrelated matrix run by about 2x
    on this box, so the load is recorded beside the timings.

    This reads the kernel's own idle/kernel/user tick counters through
    GetSystemTimes rather than summing per-process CPU. The per-process version
    tried first reported an empty list on a machine independently measured at
    1.8 busy cores, because processes the caller cannot open report no CPU time
    at all -- a load sampler that silently reads zero is worse than none, since
    it certifies quiet.
    """
    import ctypes
    from ctypes import wintypes

    if sys.platform != "win32":
        try:
            before = os.times()
            time.sleep(seconds)
            after = os.times()
            busy = (after.system - before.system) + (after.user - before.user)
            return {"busy_cores": round(busy / seconds, 3), "sampled_seconds": seconds}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    class FILETIME(ctypes.Structure):
        _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

        @property
        def value(self) -> int:
            return (self.high << 32) | self.low

    get_system_times = ctypes.windll.kernel32.GetSystemTimes

    def sample():
        idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
        if not get_system_times(ctypes.byref(idle), ctypes.byref(kernel),
                                ctypes.byref(user)):
            raise OSError("GetSystemTimes failed")
        # kernel time includes idle time; busy is everything else.
        return idle.value, kernel.value + user.value

    try:
        idle0, total0 = sample()
        time.sleep(seconds)
        idle1, total1 = sample()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    total = total1 - total0
    idle = idle1 - idle0
    cpus = os.cpu_count() or 1
    if total <= 0:
        return {"error": "no elapsed CPU time"}
    busy_fraction = 1.0 - (idle / total)
    return {
        "busy_cores": round(busy_fraction * cpus, 3),
        "busy_fraction": round(busy_fraction, 4),
        "cpu_count": cpus,
        "sampled_seconds": seconds,
    }


def run_cell(rung: str, precision: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-u", str(BENCH), "--single", rung, "opencl", precision],
        capture_output=True, text=True, env=os.environ.copy(),
    )
    if proc.returncode != 0:
        return {"rung": rung, "requested_precision": precision,
                "error": proc.stderr[-2000:]}
    lines = [line for line in proc.stdout.strip().splitlines() if line.startswith("{")]
    cell = json.loads(lines[-1])
    cell["rung"] = rung
    # PoCL writes its build failures to stderr, not through the OpenCL API, so
    # the only place they exist is here.
    cell["stderr_tail"] = proc.stderr[-800:]
    return cell


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(HERE))
    args = parser.parse_args(argv)

    sys.path.insert(0, str(B0))
    from bench_b0 import describe_environment  # noqa: E402

    print("sampling background load...", flush=True)
    load_before = background_load()

    cells = []
    for rung in RUNGS:
        for precision in PRECISIONS:
            print(f"[{rung} opencl {precision:6s}] running...", flush=True)
            t0 = time.perf_counter()
            cell = run_cell(rung, precision)
            cells.append(cell)
            if "error" in cell:
                print(f"  FAILED after {time.perf_counter()-t0:.1f}s", flush=True)
                continue
            warm = cell["runs"][1]
            print(
                f"  warm assembly {warm['assembly_s']:8.2f}s"
                f"   on-axis {cell.get('on_axis_spl_db')}"
                f"   |p| {cell.get('surface_pressure_l2')}",
                flush=True,
            )

    bundle = {
        "benchmark": "B0 assembly ladder, PoCL arm",
        "opencl_runtime_selector": os.environ.get("BEMPP_CPU_DRIVER"),
        "environment": describe_environment(),
        "background_load_before": load_before,
        "background_load_after": background_load(),
        "ladder": json.loads((B0 / "ladder" / "ladder.json").read_text(encoding="utf-8")),
        "cells": cells,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULTS-pocl.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print("wrote RESULTS-pocl.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
