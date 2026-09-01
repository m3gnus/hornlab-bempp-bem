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
| 1,338  | 2.21 s  | 0.58 s  | 3.8x  |
| 2,912  | 15.36 s | 2.15 s  | 7.1x  |
| 4,992  | 67.90 s | 5.85 s  | 11.6x |

## Warm assembly, single precision

| P1 DOF | Numba   | OpenCL  | ratio |
| ------ | ------- | ------- | ----- |
| 1,338  | 2.40 s  | 0.38 s  | 6.3x  |
| 2,912  | 12.27 s | 1.23 s  | 10.0x |
| 4,992  | 58.07 s | 3.08 s  | 18.9x |

The gap widens with mesh size in both precisions, and is consistently wider in
fp32 — OpenCL compiles a `vec8` kernel for single precision against `vec4` for
double, so it gains from the narrower type where Numba gains much less.

One correction to an earlier reading of this benchmark: **Numba is not
insensitive to fp32.** It gets meaningfully faster at the two larger rungs
(15.36 → 12.27 s and 67.90 → 58.07 s). It simply gains less than OpenCL does,
which is why the ratio grows rather than holds.

## What the ratios do not include

| P1 DOF | phase | Numba fp64 | OpenCL fp64 |
| ------ | ----- | ---------- | ----------- |
| 4,992  | linear solve | 8.05 s | 8.36 s |
| 4,992  | field evaluation | 0.416 s | 0.009 s |

The linear solve is the same SciPy LU either way and is backend-independent, as
it should be — that equality is the sanity check that the two arms really are
solving the same problem. So these are **assembly** ratios, not end-to-end solve
ratios: at 4,992 DOF the whole-solve wall time is 76.6 s against 14.3 s, a 5.4x
end-to-end difference rather than 11.6x.

Field evaluation shows the asymmetry far more sharply than assembly does — 46x
at the largest rung. That is the path this work adds a warning to.

## Threads

Assembly time against thread count, middle rung (2,912 DOF), double precision,
one fresh process per point with `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` all pinned together.

| threads | Numba assembly | Numba CPU/wall | OpenCL assembly | OpenCL CPU/wall |
| ------- | -------------- | -------------- | --------------- | --------------- |
| 1       | 45.86 s        | 1.00           | 2.18 s          | 6.02            |
| 2       | 27.60 s        | 1.88           | 2.20 s          | 5.99            |
| 4       | 17.78 s        | 3.49           | 2.24 s          | 6.09            |
| 8       | 14.49 s        | 6.63           | 2.19 s          | 5.92            |
| 12      | 15.00 s        | 9.54           | 2.20 s          | 6.06            |

Two things fall out of this, and the second is a caveat rather than a result.

**Numba stops scaling at 8 threads and then goes backwards.** From 1 to 12
threads it gains 3.1x on 12 vCPUs, and the 12-thread point is *slower* than the
8-thread point (15.00 s against 14.49 s) while consuming half again as much CPU
(9.54 against 6.63). Past 8 threads it is burning cores for nothing. So the
OpenCL advantage is not something more threads can close: at its own best
setting Numba still takes 14.49 s against OpenCL's 2.19 s.

**The OpenCL column is flat because the runtime ignores these variables, not
because it is thread-insensitive.** `NUMBA_NUM_THREADS`, `OMP_NUM_THREADS`,
`MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` do not reach the Intel CPU OpenCL
runtime, which sizes its own pool — CPU/wall sits at ~6.0 whatever they are set
to. This row is therefore evidence that **the OpenCL arm could not be
throttled by this probe**, and it is not a scaling curve. Constraining it would
need a device fission or a runtime-specific control, which was not attempted.

The useful comparison is at matched CPU consumption: Numba at 8 threads
(CPU/wall 6.63) against OpenCL as it runs (CPU/wall 6.06). At roughly equal core
usage, OpenCL is **6.6x** faster on this rung.

## Cold, and why the cold column is not a first-ever run

`cold` here means "first solve in a fresh process", which is not the same as
"first solve ever on this machine":

- **PyOpenCL keeps a persistent on-disk program cache** — 72 MB at
  `%LOCALAPPDATA%\pyopencl` on this box — shared by every process. So the
  OpenCL cold column reflects a populated kernel cache and *understates* a
  genuine first run. The one-off cost of a truly cold kernel build was measured
  separately at roughly 10 s (visible as `lhs_materialization_s` of 9.8 s on a
  cleared cache).
- Numba's cold column is dominated by LLVM compilation: 34.77 s against a warm
  2.21 s at the smallest rung, i.e. about 32 s of pure JIT.

Both are reported in `RESULTS-bundle.json`; neither belongs in a steady-state
ratio, which is why the tables above use warm times.

## Peak RSS

Whole-process peak working set, so it includes the interpreter and both
backends' imports.

| P1 DOF | Numba fp64 | OpenCL fp64 |
| ------ | ---------- | ----------- |
| 1,338  | 713 MB     | 652 MB      |
| 2,912  | 1,156 MB   | 1,262 MB    |
| 4,992  | 2,224 MB   | 2,618 MB    |

OpenCL costs somewhat more memory at the larger rungs. It is not a reason to
prefer Numba at these sizes, but it is worth knowing for a memory-bound host.

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
