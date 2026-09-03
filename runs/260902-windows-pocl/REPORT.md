# PoCL on Windows: it installs, it enumerates, and it computes nothing

PoCL was worth measuring because it is open source and would, if it landed near
the Intel CPU runtime, remove a licence question rather than answer one:
`intel-opencl-rt` ships under the Intel End User License Agreement for
Developer Tools, and bundling it into a HornLab installer needs a human read of
that EULA against this package's AGPL-3.

**It does not remove the question. It moves it somewhere worse.** PoCL's
Windows build compiles each OpenCL kernel at runtime and then needs an external
linker to turn the compiled object into a loadable module. That linker, and the
C runtime import libraries it needs, come from Microsoft Visual C++ Build
Tools. Without them every kernel build fails, and the failure is silent:
bempp-cl does not read back the build status, so the output buffer is returned
exactly as it was allocated. All zeros. Finite, correctly shaped, and faster
than any working backend.

## The headline numbers, and why none of them are a benchmark

Warm assembly on the B0 ladder: same meshes, same 2 kHz single frequency, same
process-per-cell shape as `runs/260901-windows-b0`. The Numba and Intel columns
are that run's, unchanged. The last two columns are the fingerprint.

| rung | P1 DOF | precision | Numba | Intel | PoCL | PoCL surface norm | PoCL on-axis |
| ---- | ------ | --------- | ------- | ------ | ------ | ----------------- | ------------ |
| L1 | 1,338 | double | 2.07 s | 0.58 s | 1.09 s | 0.0 | 0.0 dB |
| L1 | 1,338 | single | 2.23 s | 0.37 s | 1.15 s | 0.0 | 0.0 dB |
| L2 | 2,912 | double | 15.24 s | 2.25 s | 1.18 s | 0.0 | 0.0 dB |
| L2 | 2,912 | single | 11.95 s | 1.24 s | 1.25 s | 0.0 | 0.0 dB |
| L3 | 4,992 | double | 66.51 s | 5.83 s | 1.43 s | 0.0 | 0.0 dB |
| L3 | 4,992 | single | 57.03 s | 3.11 s | 1.38 s | 0.0 | 0.0 dB |

Read as a benchmark, PoCL is **4.1x faster than the Intel runtime and 47x
faster than Numba** at the largest rung. Every one of those seconds is spent
not assembling a matrix.

The tell is in the shape rather than the size. Dense assembly is O(N^2), so
across a 3.7x DOF range Numba grows 32x and Intel grows 10x. PoCL grows
**1.3x** — it is flat, because what is being timed is process start, mesh load
and buffer allocation. A column that does not scale with the problem is not
measuring the problem.

That is why the fingerprint columns exist. `bench_b0.py` declared
`on_axis_spl_db` and left it `None`; this run fills it in, along with the L2
norm of the surface pressure trace, because a timing column cannot tell "fast"
from "did nothing".

## Where it fails, exactly

`scripts/windows_smoke.py` against PoCL 7.0.0, by stage:

| stage | result |
| ----- | ------ |
| `pyopencl` | passes — imports, enumerates two platforms |
| `device` | passes — `cpu`, 12 compute units, `double_fp_config` 191 |
| `assembly` | **passes, wrongly** — `check_opencl` reported `ok` |
| solve on numba | passes, 45.92 s |
| solve on opencl | **passes, wrongly** — 6.12 s, reported as "7.50x numba" |
| agreement | **fails** — relative difference `1.000e+00` against tolerance `1e-08` |

So the smoke script does catch it, and only at its last check. Everything
before that reported a healthy machine, and the one number a reader would have
quoted from the run — 7.50x — is an artefact of returning zeros.

The underlying error never reaches the OpenCL API. It goes to stderr and
nowhere else:

```
warning: unable to find a Visual Studio installation; try running Clang from a developer command prompt
error: unable to execute command: program not executable
error: linker command failed with exit code 1 (use -v to see invocation)
clWaitForEvents failed with code -14
```

With a linker made available, the requirement becomes explicit:

```
link: error: could not open 'libcmt.lib': no such file or directory
link: error: could not open 'oldnames.lib': no such file or directory
```

`libcmt.lib` and `oldnames.lib` are the MSVC C runtime import libraries. They
are not part of the Windows SDK, and they are not something an audio
application can reasonably ask a user to install.

## What was tried, so nobody repeats it

A third-party build and the vendor's own build fail identically, so this is not
a packaging accident.

| attempt | result |
| ------- | ------ |
| conda-forge `pocl` 7.1 win-64 | platform loads, **zero devices** — its `pocl.dll` imports `zstd.dll`, which nothing in the package's dependency closure provides. Installing `zstd` fixes enumeration. |
| conda-forge `pocl` 7.1, devices up | CPU device present, kernel build fails at link |
| upstream `PoCL-7.0.0-CONF-win64.exe` | installs cleanly, ships per-ISA kernel bitcode including AVX2, CPU device present, **same link failure** |
| LLVM `lld-link` on `PATH` | not used — clang's MSVC driver never looks there once Visual Studio detection has failed |
| `lld-link` presented as the MSVC toolchain linker | clang invokes it; it then fails for want of `libcmt.lib` and `oldnames.lib` |

PoCL 7.1 has no upstream Windows binary at all — that release carries only
signatures. 7.0.0 and 7.2-RC1 do.

**PoCL is not broken in general.** This repository's own CI installs
`pocl-opencl-icd` on its Linux leg and gets 392 passed against a real device.
On Linux the kernel link uses the system toolchain, which is always there. The
problem is specific to Windows, where the equivalent toolchain is not.

## The guard this exposed, and the fix

`check_opencl()` exists to catch a runtime that is present and enumerable but
wrong, and its docstring says so. It did not catch this one: it validated the
assembled operator's shape and finiteness, and an all-zero matrix is both.

`hornlab_bempp_bem/device.py` now also rejects an operator that is entirely
zero, or that has a zero anywhere on its diagonal — the single-layer
self-interaction terms are strongly singular, so a zero there is never
physical. Against PoCL it now reports:

```
OpenCL UNAVAILABLE at stage 'assembly': OpenCLError: assembled operator is
entirely zero, so the OpenCL kernel did not run.
```

and against the Intel runtime it still reports `ok`.

The existing test for a broken runtime modelled the failure as one that
*raises*. Real ones do not, which is exactly why the hole was there. Two tests
now model the silent shapes: an all-zero operator, and one with a hole in its
diagonal.

What this still cannot catch is a runtime that computes non-zero but wrong
values. Nothing self-contained can — that needs a second backend to compare
against, which is what `windows_smoke.py` does, and why it exists alongside
`check_opencl` rather than being replaced by it.

## This reaches Waveguide Generator

`server/solver/bempp.py`'s `_opencl_status()` is the probe the shipped app's
capability report and solve path consult. It stops at "a CPU-type device exists
and `default_cpu_device()` initialised". Run against PoCL on this box it
returns, verbatim:

```json
{"available": true,
 "reason": "bempp-cl assembles on OpenCL device cpu (Portable Computing Language)",
 "backend": "opencl", "fallback": null}
```

The app would report READY, choose OpenCL, decline to fall back to Numba, and
return silence from every solve. The reason string asserts an assembly that
never happened. `scripts/check_backends.py` already warns in its own docstring
that "a device that initialises can still fail to compile the assembly kernel"
— the awareness is in the comment and not in the code.

That is a Waveguide Generator change and is not made here.

## Recommendation

**The EULA question still needs answering.** PoCL does not dissolve it. The
choice it actually offers is between shipping one proprietary Intel DLL and
requiring every user to install Microsoft Visual C++ Build Tools, which is a
larger licence surface, a multi-gigabyte prerequisite, and a support burden on
a machine that will otherwise look like it is working.

Two decisions are easy to conflate here and are worth keeping apart:

- **Per-machine installation** of `intel-opencl-rt` by a developer, which is
  what this box does, needs no read of the EULA by anyone but that developer.
- **Bundling it into a HornLab installer** is redistribution, and that is the
  question for Magnus. This work does not change it.

Revisit PoCL if upstream ships a Windows build that links kernels without an
MSVC toolchain — a statically linked `lld` plus a freestanding kernel ABI would
do it. Until then its failure mode is worse than its absence: a user with PoCL
installed gets a confident READY and silent output, where a user with no
runtime at all gets a correct, slow Numba fallback.

## Caveats on this machine

- **QEMU VM, AMD Ryzen 7 5825U, 12 vCPUs, AVX2 only, `max_clock_frequency`
  reads 0.** Same box and the same caveats as `runs/260901-windows-b0`.
- **Background load was 1.6 busy cores of 12** during the run (Windows Search
  indexing), recorded in `RESULTS-pocl.json` at sample time. It affects no
  conclusion here: the PoCL column measures a no-op, and the Numba and Intel
  columns are quoted from the earlier separate run rather than re-measured
  against it.
- **UAC is disabled on this host (`EnableLUA=0`).** Every process runs at High
  integrity, and the Khronos ICD loader deliberately refuses to read
  `OCL_ICD_FILENAMES` when it is. That is why PoCL had to be reached through an
  HKLM ICD registration rather than an environment variable, and it is worth
  knowing before trusting anything measured on this box about Windows security
  behaviour.

## Reproducing

```
PoCL-7.0.0-CONF-win64.exe /S /D=C:\pocl700
set BEMPP_CPU_DRIVER=Portable
python scripts/windows_smoke.py --json smoke-pocl.json
python runs/260902-windows-pocl/bench_pocl.py --out runs/260902-windows-pocl
```

The installer registers its own ICD. Intel remains vendor value 0 and stays the
default for any process that does not ask for a different one, so adding PoCL
does not change what anything else on the machine selects.

| file | contents |
| ---- | -------- |
| `bench_pocl.py` | drives `bench_b0.py`'s OpenCL arm against a selected runtime |
| `RESULTS-pocl.json` | every cell, cold and warm, fingerprints, and the background load at sample time |
