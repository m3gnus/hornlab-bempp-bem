# B0 — Numba vs OpenCL CPU assembly on Windows

What a missing CPU OpenCL runtime costs, measured rather than assumed. bempp-cl
falls back from OpenCL to Numba silently, and this is the number that decides
whether that fallback deserves a warning.

## Hardware caveat — read this before quoting any ratio

**This is a QEMU virtual machine, not bare metal.** Specifically:

- 12 vCPUs. The host topology is not visible, so "cores" below means vCPUs and
  nothing here can distinguish a physical core from a sibling thread.
- **AVX2 only.** The OpenCL device reports `+avx2` and every AVX-512 feature
  absent (`-avx512f`, `-avx512bw`, `-avx512vl`, …). On an AVX-512 part both
  backends would vectorize differently and these ratios would not carry over.
- **`max_clock_frequency` reads 0.** The hypervisor does not expose the real
  clock, so per-core throughput cannot be normalized and no absolute
  seconds-per-DOF claim from this box is portable.

The ratios are internally consistent — same box, same meshes, same process
shape, both backends — so they support "OpenCL is N times faster *here*". They
do not support a claim about any other machine, and least of all about bare
metal with AVX-512.

## Method

- Geometry: ASRO68, remeshed from ATH's own `bem_mesh.geo` into a three-rung
  P1-DOF ladder by `make_ladder.py`. Rungs landed at **1,338 / 2,912 / 4,992**
  P1 DOF (targets were 1,364 / 2,884 / 4,984).
- The ladder is **welded**: gmsh emits one node per (surface, position) pair, so
  its classification left 83/157/203 duplicate nodes on the seams. This package
  welds them on load, so its DOF count was already correct, but the file itself
  was not watertight and a consumer that does not weld saw a different mesh.
  `make_ladder.py` now welds at generation, so file nodes = P1 DOF exactly.
- The mesh is millimetres, converted with `load_mesh(path, scale=0.001)`. Note
  that `SolveConfig.mesh_scale` does **not** do this for an already-loaded mesh:
  `_resolve_mesh` returns a `LoadedMesh` untouched (a test pins that it must not
  reload), so pairing a pre-loaded mesh with `mesh_scale` silently solves a
  411 m horn instead of a 411 mm one.
- 2 kHz, one frequency, LU, driver patch excited by unit normal velocity.
- Every cell runs in its **own process**, and solves the same frequency twice.
  The first solve is `cold`, the second `warm`. All ratios below use warm only.
- `bempp-cl does not break out singular and near-singular corrections as a
  separate phase` — they run inside the layer-potential assembly kernels. No
  number is invented for them here. `assembly` is operator construction plus
  the SLP and DLP assemblies; the dense materialization is reported separately
  because on the cold path it is the largest single term.

## Warm assembly, double precision

| P1 DOF | Numba   | OpenCL  | ratio |
| ------ | ------- | ------- | ----- |
| 1,338  | 2.07 s  | 0.58 s  | 3.6x  |
| 2,912  | 15.24 s | 2.25 s  | 6.8x  |
| 4,992  | 66.51 s | 5.83 s  | 11.4x |

## Warm assembly, single precision

| P1 DOF | Numba   | OpenCL  | ratio |
| ------ | ------- | ------- | ----- |
| 1,338  | 2.23 s  | 0.37 s  | 6.0x  |
| 2,912  | 11.95 s | 1.24 s  | 9.6x  |
| 4,992  | 57.03 s | 3.11 s  | 18.3x |

A third runtime, PoCL 7.0.0, was measured against this same ladder on
2026-09-02 and is reported in `runs/260902-windows-pocl`, which carries the
combined table. Its column is not a result: PoCL cannot link a kernel on
Windows without an MSVC toolchain, so it returns all-zero matrices, and its
"assembly" times are flat at 1.1-1.4 s across the whole ladder. It is kept
because that is what a broken runtime looks like from inside this benchmark.

The gap widens with mesh size in both precisions, and is consistently wider in
fp32 — OpenCL compiles a `vec8` kernel for single precision against `vec4` for
double, so it gains from the narrower type where Numba gains much less.

One correction to an earlier reading of this benchmark: **Numba is not
insensitive to fp32.** It gets meaningfully faster at the two larger rungs
(15.24 -> 11.95 s and 66.51 -> 57.03 s). It simply gains less than OpenCL does,
which is why the ratio grows rather than holds.

## What the ratios do not include

| P1 DOF | phase | Numba fp64 | OpenCL fp64 |
| ------ | ----- | ---------- | ----------- |
| 4,992  | linear solve | 8.32 s | 8.31 s |
| 4,992  | field evaluation | 0.394 s | 0.009 s |

The linear solve is the same SciPy LU either way and is backend-independent, as
it should be — that equality is the sanity check that the two arms really are
solving the same problem. So these are **assembly** ratios, not end-to-end solve
ratios: at 4,992 DOF the whole-solve wall time is 75.5 s against 14.2 s, a 5.3x
end-to-end difference rather than 11.4x.

Field evaluation shows the asymmetry far more sharply than assembly does — 44x
at the largest rung. That is the path this work adds a warning to.

## Threads

Assembly time against thread count, middle rung (2,912 DOF), double precision,
one fresh process per point with `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` all pinned together.

| threads | Numba assembly | Numba CPU/wall | OpenCL assembly | OpenCL CPU/wall |
| ------- | -------------- | -------------- | --------------- | --------------- |
| 1       | 38.30 s        | 1.00           | 2.24 s          | 5.93            |
| 2       | 22.03 s        | 1.87           | 2.22 s          | 6.00            |
| 4       | 14.83 s        | 3.46           | 2.21 s          | 6.09            |
| 8       | 14.06 s        | 6.61           | 2.25 s          | 6.05            |
| 12      | 15.12 s        | 9.10           | 2.19 s          | 6.08            |

Two things fall out of this, and the second is a caveat rather than a result.

**Numba stops scaling at 8 threads and then goes backwards.** From 1 to 12
threads it gains 2.5x on 12 vCPUs, and the 12-thread point is *slower* than the
8-thread point (15.12 s against 14.06 s) while consuming half again as much CPU
(9.10 against 6.61). Past 8 threads it is burning cores for nothing. So the
OpenCL advantage is not something more threads can close: at its own best
setting Numba still takes 14.06 s against OpenCL's 2.25 s.

**The OpenCL column is flat because the runtime ignores these variables, not
because it is thread-insensitive.** `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` do not reach the Intel CPU OpenCL
runtime, which sizes its own pool — CPU/wall sits at ~6.0 whatever they are set
to. This row is therefore evidence that **the OpenCL arm could not be
throttled by this probe**, and it is not a scaling curve. Constraining it would
need a device fission or a runtime-specific control, which was not attempted.

The useful comparison is at matched CPU consumption: Numba at 8 threads
(CPU/wall 6.61) against OpenCL as it runs (CPU/wall 6.05). At roughly equal core
usage, OpenCL is **6.2x** faster on this rung.

## Cold, and why the cold column is not a first-ever run

`cold` here means "first solve in a fresh process", which is not the same as
"first solve ever on this machine":

- **PyOpenCL keeps a persistent on-disk program cache** — 72 MB at
  `%LOCALAPPDATA%\pyopencl` on this box — shared by every process. So the
  OpenCL cold column reflects a populated kernel cache and *understates* a
  genuine first run. The one-off cost of a truly cold kernel build was measured
  separately at roughly 10 s (visible as `lhs_materialization_s` of 9.8 s on a
  cleared cache).
- Numba's cold column is dominated by LLVM compilation: 34.28 s against a warm
  2.07 s at the smallest rung, i.e. about 32 s of pure JIT.

Both are reported in `RESULTS-bundle.json`; neither belongs in a steady-state
ratio, which is why the tables above use warm times.

## Peak RSS

Whole-process peak working set, so it includes the interpreter and both
backends' imports.

| P1 DOF | Numba fp64 | OpenCL fp64 |
| ------ | ---------- | ----------- |
| 1,338  | 712 MB     | 653 MB      |
| 2,912  | 1,158 MB   | 1,116 MB    |
| 4,992  | 2,227 MB   | 2,466 MB    |

OpenCL costs somewhat more memory at the largest rung and slightly less at the
middle one. It is not a reason to prefer either at these sizes, but it is worth
knowing for a memory-bound host.

## Files

| file | contents |
| ---- | -------- |
| `make_ladder.py` | builds the three-rung ladder from ATH's `bem_mesh.geo`; needs `HORNLAB_ASRO68_GEO` |
| `bench_b0.py` | the benchmark; `--threads-probe` for the thread arm, `--single` for one cell |
| `ladder/ladder.json` | the rung manifest: DOF, triangle and driver counts, and the mesh-size factor that produced each |

**The ladder `.msh` files are deliberately not committed.** This repository is
public and tracks no mesh at all — `tests/test_reference_asro68.py` reaches its
reference through `HORNLAB_ASRO68_MESH` for the same reason. The rungs are
remeshed ASRO68, so committing them would publish the horn geometry. Regenerate
them with `make_ladder.py`; `ladder.json` pins the mesh-size factors, so the
result is reproducible from the same `.geo`.
| `RESULTS-bundle.json` | every cell, cold and warm, all phase timings verbatim |
| `L3_warm.json` | the largest rung split into cold and warm |
| `threads_probe.json` | the thread-scaling arm |

## Reproducing

```bash
export HORNLAB_ASRO68_GEO=/path/to/250917asro68/ABEC_FreeStanding/bem_mesh.geo
python make_ladder.py
python bench_b0.py --out .
python bench_b0.py --threads-probe
```

`NUMBA_CACHE_DIR` should point somewhere short: Numba's cache path can cross
Windows' 260-character `MAX_PATH`, and it fails as a `FileNotFoundError` from
inside `numba.core.caching` rather than as anything that mentions path length.
