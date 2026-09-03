# B6 — BEAT CPU against bempp-OpenCL

Same box, same three-rung ASRO68 ladder, same excitation, three engines:
`bempp-fp64`, `bempp-fp32` (both OpenCL) and `beat-cpu`
(`HORNLAB_BEAT_FORCE_CPU=1`).

## Hardware caveat

Same box as B0, and the same limits apply: a QEMU VM, 12 vCPUs, **AVX2 only**,
`max_clock_frequency` reads 0. Ratios here are internally consistent and are not
portable to bare metal.

## What each engine actually did

Recorded rather than assumed, because two of these differ and both affect the
numbers below.

| | bempp | BEAT CPU |
| --- | --- | --- |
| solver | SciPy LU (dense direct) | "Julia direct dense solve" |
| iterations | none (direct) | none (direct) |
| precision | float64 / float32 as requested | float64 |
| regular quadrature | fixed order | **wavelength-adaptive**, order 2 with a k·h window |
| BLAS threads | unrestricted (12) | **4** |
| symmetry | off | off |

The quadrature difference is the important one. BEAT raises quadrature order with
k·h, so its assembly cost climbs with frequency; bempp on this base does not, so
its assembly cost is flat in frequency. Any ratio between them is therefore a
function of frequency, which is why both 2 kHz and 20 kHz are reported and why
neither alone would be honest.

**The linear solver was forced to LU on the bempp side.** Left at its default
this package selects GMRES; BEAT's default is a direct dense solve, so LU is the
like-for-like choice. (An earlier run at bempp's default did not converge, but
that run also had the scale fault described below, so no claim is made here
about the default solver's convergence — it needs its own measurement.)

## Two faults found before these numbers were trusted

Both were in the harness, not in either engine, and both produced a *large*
apparent engine disagreement:

1. **Mesh scale.** `SolveConfig.mesh_scale` is inert when the caller passes an
   already-loaded `LoadedMesh` — `_resolve_mesh` returns it untouched, and a
   test pins that it must not reload. The conversion belongs on
   `load_mesh(path, scale=...)`. BEAT takes a *path*, so it needs `mesh_scale`
   and bempp needs `load_mesh(scale=)`: the requirement sits in opposite places
   in the two APIs. Getting either wrong silently solves a 411 m horn.
2. **Unwelded meshes.** gmsh emits one node per (surface, position) pair, so the
   ladder carried 83/157/203 duplicate nodes on the seams. bempp welds on load
   and saw 1,338 vertices; BEAT does not and saw 1,421 — **the two engines were
   solving different discretizations**, BEAT's with a P1 space discontinuous
   across every seam.

Together these produced a fake 90–100 dB on-axis disagreement. The harness now
computes the driver-patch area from the mesh file and checks what each engine
reports against it, so a scale fault shows up as a factor of ~10⁶ in
`geometry_check_passed` rather than as a puzzling far field. Every cell below
passes that check and both engines report 1,338 / 2,912 / 4,992 vertices.

## Ratio table — warm wall-clock seconds

Warm means the second solve of the same frequency in the same process. Cold is
reported separately below because for BEAT it is dominated by Julia's JIT.

| rung | P1 DOF | freq | bempp-fp64 | bempp-fp32 | BEAT CPU | BEAT / fp64 |
| ---- | ------ | ---- | ---------- | ---------- | -------- | ----------- |
| L1 | 1,338 | 2 kHz  | 0.86 s | 0.65 s | 1.36 s  | 1.58x |
| L1 | 1,338 | 20 kHz | 0.90 s | 0.69 s | 2.70 s  | 3.00x |
| L2 | 2,912 | 2 kHz  | 4.41 s | 3.25 s | 4.80 s  | 1.09x |
| L2 | 2,912 | 20 kHz | 4.12 s | 3.17 s | 9.82 s  | 2.38x |
| L3 | 4,992 | 2 kHz  | 14.32 s | 11.04 s | 13.23 s | **0.92x** |
| L3 | 4,992 | 20 kHz | 14.26 s | 10.90 s | 28.25 s | 1.98x |

BEAT is competitive with bempp-OpenCL at 2 kHz and overtakes it at the largest
rung, but is 2–3x slower at 20 kHz. That crossover is the adaptive quadrature:
at high k·h BEAT is doing more work per element on purpose.

Split by phase, at L3:

| phase | bempp-fp64 2 kHz | BEAT 2 kHz | bempp-fp64 20 kHz | BEAT 20 kHz |
| ----- | ---------------- | ---------- | ----------------- | ----------- |
| assembly | 5.89 s | 10.95 s | 5.82 s | 25.92 s |
| linear solve | 8.35 s | **1.55 s** | 8.37 s | 1.72 s |
| field evaluation | 0.008 s | 0.008 s | 0.009 s | 0.020 s |

The two engines are strong in different places. bempp assembles faster;
**BEAT's dense solve is 5x faster** (1.55 s against 8.35 s at 4,992 DOF), and it
achieves that on 4 BLAS threads against bempp's 12. That is the single largest
per-phase difference in this table and it is what makes BEAT competitive overall
despite the slower assembly.

Neither engine reports singular and near-singular corrections as a separate
phase — both fold them into assembly — so no such column exists here. Inventing
one would have meant inventing the number.

### CPU/wall and peak memory

Measured across the whole process tree with a Windows job object, because BEAT's
solve runs in a Julia child process; `process_time()` reports ~0.00 for it and
the parent's peak RSS misses the solve entirely.

| rung | bempp-fp64 CPU/wall | BEAT CPU/wall | bempp-fp64 peak | BEAT peak |
| ---- | ------------------- | ------------- | --------------- | --------- |
| L1 | 6.8 | 8.1  | 1,366 MB | 3,377 MB |
| L2 | 5.7 | 9.0  | 1,860 MB | 4,268 MB |
| L3 | 5.1 | 9.8  | 3,211 MB | 6,521 MB |

BEAT uses roughly twice the memory and roughly twice the cores.

### Cold, and Julia's JIT

| rung / freq | bempp-fp64 cold | warm | BEAT cold | warm | BEAT JIT |
| ----------- | --------------- | ---- | --------- | ---- | -------- |
| L1 / 2 kHz | 14.09 s | 0.86 s | 25.75 s | 1.36 s | **~24.4 s** |
| L3 / 2 kHz | 26.74 s | 14.32 s | 37.39 s | 13.23 s | ~24.2 s |

BEAT pays a flat ~24 s of Julia JIT on the first frequency in a fresh process,
independent of mesh size — the same trap as Numba's LLVM warm-up in B0, and the
reason every cell here solves twice. bempp's cold overhead is ~10–13 s of OpenCL
kernel build against an already-populated on-disk cache.

## Agreement — far field only

Scored against `bempp-fp64`. Patterns are normalized to each engine's own
on-axis level first, so this is **shape**, not level; absolute SPL equality and
null depth are deliberately not compared.

| rung | freq | candidate | main-lobe rms 0–60° | max abs | −6 dB BW Δ | on-axis mag | on-axis phase |
| ---- | ---- | --------- | ------------------- | ------- | ---------- | ----------- | ------------- |
| L1 | 2 kHz | bempp-fp32 | 0.000 dB | 0.001 dB | −0.00° | 0.000 dB | −0.00° |
| L1 | 2 kHz | BEAT | 1.511 dB | 3.126 dB | +19.57° | −4.885 dB | −14.81° |
| L2 | 2 kHz | bempp-fp32 | 0.000 dB | 0.000 dB | 0.00° | −0.000 dB | 0.00° |
| L2 | 2 kHz | BEAT | **0.049 dB** | 0.088 dB | +0.43° | −0.058 dB | +5.80° |
| L3 | 2 kHz | bempp-fp32 | 0.000 dB | 0.000 dB | 0.00° | 0.000 dB | 0.00° |
| L3 | 2 kHz | BEAT | **0.018 dB** | 0.038 dB | +0.13° | −0.060 dB | +6.20° |
| L1 | 20 kHz | BEAT | 10.683 dB | 17.532 dB | −14.10° | +11.772 dB | +89.86° |
| L2 | 20 kHz | BEAT | 1.523 dB | 3.811 dB | −2.62° | +1.946 dB | +65.42° |
| L3 | 20 kHz | BEAT | 1.148 dB | 2.502 dB | −0.40° | +0.354 dB | +64.52° |

**The engines converge on each other as the mesh refines.** At 2 kHz the
main-lobe rms falls 1.511 → 0.049 → 0.018 dB across the ladder and the beamwidth
difference falls 19.6° → 0.43° → 0.13°. At the best-resolved point (L3, 2 kHz,
20.5 elements per wavelength) the two independent implementations agree to
**0.018 dB rms over the main lobe, 0.13° of beamwidth and 0.06 dB on axis**.
That is the number worth quoting.

`bempp-fp32` against `bempp-fp64` is 0.000 dB everywhere — single precision costs
nothing measurable on the far field at these sizes, while running ~25% faster.

### The 20 kHz rows are not an accuracy statement

This ladder cannot resolve 20 kHz. Wavelength is 17.15 mm against mean element
edges of 18.25 / 11.41 / 8.38 mm, i.e. **0.9 / 1.5 / 2.0 elements per
wavelength**, against roughly 6 as a working minimum. Both engines solve the same
discrete problem there, so engine-vs-engine agreement remains a meaningful check
on the implementations, and it still improves monotonically with refinement
(10.68 → 1.52 → 1.15 dB). Neither is near the physical answer. At 2 kHz the
ladder gives 9.4 / 15.0 / 20.5 elements per wavelength and is sound.

### Phase: a constant delay, not a sign flip

The on-axis phase difference is a consistent *positive* offset that scales with
frequency:

| rung | 2 kHz | 20 kHz | implied delay |
| ---- | ----- | ------ | ------------- |
| L2 | +5.80° | +65.42° | 8.06 µs / 9.09 µs |
| L3 | +6.20° | +64.52° | 8.61 µs / 8.96 µs |

φ/(360·f) is ~8–9 µs at both frequencies, i.e. a fixed time offset of roughly
3 mm of path — not the delay-sign disagreement that was anticipated. Recorded,
not reconciled.

### Impedance agrees, contrary to expectation

The two engines were expected to disagree here because the outputs are different
quantities. On this model they do not. At L2, 2 kHz:

| engine | Z |
| ------ | - |
| bempp-fp64 | +0.013543 +0.0130389j |
| bempp-fp32 | +0.013543 +0.0130389j |
| BEAT CPU | +0.013601 +0.0131006j |

That is 0.4% apart in both parts. Recorded as measured; it does not license
treating the two impedance outputs as interchangeable in general.

## 12-point log sweep, 100 Hz – 20 kHz, middle rung

One process per engine, frequencies in order — the way a sweep actually runs, so
compilation is paid once rather than twelve times.

| engine | total | CPU/wall | first point | warm mean |
| ------ | ----- | -------- | ----------- | --------- |
| bempp-fp64 | 68.41 s | 4.66 | 13.44 s | 3.92 s |
| bempp-fp32 | 59.33 s | 3.04 | 12.96 s | 2.98 s |
| BEAT CPU | 87.16 s | 7.39 | 7.86 s | 5.34 s |

Per point, in seconds (100 Hz first, 20 kHz last):

```
bempp-fp64  13.44  4.18  4.09  3.77  3.78  3.77  3.93  3.97  3.96  3.98  3.84  3.80
bempp-fp32  12.96  2.99  3.01  2.96  2.99  3.00  3.02  3.00  3.01  3.02  2.84  2.98
beat-cpu     7.86  4.37  4.44  4.42  4.45  4.49  4.37  4.34  4.40  4.32  9.60  9.59
```

The shape is the whole story: **bempp is flat in frequency, BEAT is not.** BEAT
is faster than bempp-fp64 across the bottom nine points and then roughly doubles
at 12.4 kHz and 20 kHz as its adaptive quadrature raises order. Over the whole
sweep bempp-fp64 wins on wall time (68 s against 87 s) purely because of those
last two points.

Note that the sweep's first point is 100 Hz and therefore cheap, so its
cold-minus-warm gap is *not* a clean JIT measurement. The spot cells above, which
solve the same frequency twice, are.

## Coupled

**BEAT's coupled CPU solver is not reachable through the Python wrapper**:
`SolveConfig` exposes no coupled, chamber or interior field and the wrapper
contains no `coupled` reference, though `BeatEngineCoupled.jl` exists on the
Julia side. No coupled row is reported.

## Files

| file | contents |
| ---- | -------- |
| `bench_b6.py` | the benchmark; `--single` for one cell, `--only-sweep` for the sweep arm |
| `RESULTS-b6.json` | every cell, cold and warm, phase timings verbatim, far-field pressure, geometry check |
| `sweep_L2.json` | the 12-point sweep, per-frequency |

The ladder itself lives in `../260901-windows-b0/ladder/` and its `.msh` files
are deliberately not committed — see that report.

## Reproducing

```bash
export HORNLAB_BEAT_FORCE_CPU=1
python bench_b6.py --out . --julia /path/to/julia
```
