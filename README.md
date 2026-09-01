# hornlab-bempp-bem

Cross-platform `bempp-cl` acoustic BEM solver for HornLab waveguide and
loudspeaker surface meshes.

This is the Bempp sibling of `hornlab-metal-bem`. Use this package when a
portable Python/OpenCL/Numba solver path is more appropriate than the
Apple-Silicon-only Metal backend.

Use the `hornlab_bempp_bem` namespace for new integrations.

## Status

The package wraps HornLab's canonical Bempp solve path for Gmsh triangle
surface meshes. It supports standard Neumann solves, complex-k shifted solves,
Burton-Miller solves, optional Robin wall admittance through
`impedance_sources`, native half/quarter mirror symmetry for rigid Neumann
models, and serial or parallel frequency sweeps.

`bempp-cl` supplies the numerical assembly and potential evaluation backend.
The package does not import `gmsh` at runtime; meshes are read through
`meshio`.

## Quick Start

Run a solve from a Gmsh `.msh` file with canonical physical tags:

```python
from hornlab_bempp_bem import SolveConfig, solve

config = SolveConfig(
    velocity_sources={2: 1.0},
    freq_min_hz=500.0,
    freq_max_hz=20_000.0,
    freq_count=40,
)

result = solve("waveguide.msh", config)

print(result.frequencies_hz.shape)
print(result.directivity_db.shape)
print(result.impedance.shape)
```

Canonical physical tags:

- `1`: rigid wall
- `2`: primary velocity source
- `3` and `4`: optional source, aperture, or model-specific tags

## Inputs

`solve(mesh, config=None)` accepts either:

- a path to a Gmsh `.msh` triangle surface mesh
- a `LoadedMesh` returned by `load_mesh()`

Mesh requirements:

- coordinates are metres unless `mesh_scale` is set
- mesh cells must contain triangles
- triangle cells must have physical-group tags
- triangle winding must be outward for canonical meshes
- source/radiator tags must match `config.velocity_sources`
- mirror-reduced meshes must occupy the positive side of each declared
  symmetry plane and set `native_symmetry_plane`

Signed-volume winding validation is applied to closed two-manifold meshes. It
is intentionally not used to flip or reject open surfaces because their signed
volume changes under a rigid translation; callers remain responsible for the
declared outward winding on open/bare meshes.

With `require_closed_mesh=True` (or `load_mesh(..., require_closed=True)`),
the surface must additionally be a closed 2-manifold: every edge shared by
exactly two triangles. Open edges and non-manifold edges are rejected. The
check also applies to pre-loaded `LoadedMesh` inputs passed to `solve()`. For
native symmetry, the check is applied after mirroring, so cut-plane edges are
allowed but any remaining leak is rejected.

On an open surface, Bempp's default continuous P1 space excludes free-boundary
degrees of freedom, constraining pressure to zero along the free edges.
`hornlab-metal-bem` instead uses an all-vertex P1 pressure space. Open-mesh
results are therefore not directly comparable between the two backends.
Closed meshes have no free-boundary degrees of freedom, so the two P1 space
choices are identical there.

Use `solve_frequencies(mesh, frequencies_hz, config=None)` when frequency
order comes from the caller instead of a generated sweep.

## Configuration

`SolveConfig` controls the solve.

Common fields:

- `freq_min_hz`, `freq_max_hz`, `freq_count`, `freq_spacing`
- `velocity_sources`, mapping physical tag to source weight
- `velocity_mode`, either `VelocityMode.ACCELERATION` or `VelocityMode.VELOCITY`
- `formulation`, one of `STANDARD`, `COMPLEX_K`, or `BURTON_MILLER`
- `slp_dlp_quadrature`, `slp_dlp_singular_quadrature`, and
  `hyp_adlp_quadrature`; singular orders 3 and 5 are accepted on Numba but
  rejected after backend resolution when the pinned bempp-cl OpenCL assembler
  would silently discard points at those orders
- `solver`, one of `GMRES` (default), `LU`, or `AUTO`; `AUTO` is an explicit
  opt-in that selects LU at or below `lu_threshold` (6000 elements by default)
  and GMRES above it. Making `AUTO` the default would require a measurement
  campaign across representative below- and above-threshold meshes and
  difficult frequencies, covering complex pressure, SPL, impedance,
  convergence, wall time, and peak memory; that qualification has not been run
- `gmres_tol` (default `1e-6`) and optional `gmres_restart`
- `observation`, an `ObservationConfig`
- `mesh_scale`
- `air_density`
- `require_closed_mesh`, reject open or non-manifold surfaces before solving
- `assembly_backend`, one of `"opencl"`, `"numba"`, or `"auto"`
- `precision`, `"single"` (default) or `"double"`; Robin impedance solves
  explicitly promote operator assembly, the direct system, and retained traces
  to double precision, and expose requested/effective precision in `solver_log`.
  AUTO falls back to Numba if the selected OpenCL device lacks fp64; explicitly
  requesting OpenCL instead raises an early error naming the limitation
- `opencl_device`, either `"cpu"` or `"gpu"` when using OpenCL
- `native_symmetry_plane`, one of `None`, `"yz"`, `"xz"`, or `"yz+xz"`
- `restrict_neumann_space`, exactly omit rigid-wall zero Neumann columns from
  boundary and potential assembly
- `return_surface_traces`, retain P1 pressure and total DP0 Neumann
  coefficients for post-solve exterior-field evaluation
- `progress_callback`
- `on_frequency_result`, for streaming progress and early stop

`on_frequency_result` stops only when it returns exactly `False`; callbacks
used only for side effects may return `None` and the sweep continues. Serial and
parallel sweeps populate the same result fields, including
`surface_pressure_avg`.

## Performance

Two bempp-cl behaviours dominate the cost of a sweep on production-sized meshes,
and both are worked around in process from `configure_opencl`:

- **bempp-cl rebuilds its OpenCL program on every operator assembly.** Nothing
  in the build key varies across a sweep -- the wavenumber reaches the kernel as
  a runtime buffer argument, never a compile-time macro -- yet each assembly
  re-enters the full pyopencl build path, about 20-25 ms per call, three calls
  per assembly and four assemblies per frequency. Memoizing it
  (`_opencl_program_cache`) leaves assembled matrices **bitwise identical** and
  cuts warm sweep time by 1.5-2.8x. The key covers the OpenCL context, so a
  rebound device is never served a stale kernel, and the cache is thread-local
  because `pyopencl.Kernel` is mutable.
- **bempp-cl's kernel include path is unquoted**, which breaks OpenCL assembly
  outright on any install path containing a space.

Separately, LAPACK's LU scales *negatively* at waveguide matrix sizes -- a
260x260 complex solve takes 1.7 ms on one BLAS thread and 29 ms on twelve --
while GEMM scales normally. `_blas_threads` limits threads around the dense
solve only, so the factorization gets its speed-up without taxing the rest of
the process. It is best-effort: an unrecognised BLAS build leaves the thread
count alone.

## Workers

`workers` selects the frequency-sweep execution mode:

- `1` (default): one process, one warm interpreter.
- `0`: auto. Splits across processes only when each worker would get at least
  40 frequencies.
- `n > 1`: exactly `n` worker processes, honoured as given, with a warning when
  the sweep is too short to be worth it.

The threshold is not arbitrary. A spawned worker re-imports bempp-cl and re-JITs
its numba kernels before it can solve anything -- about 5 s -- against roughly
0.13 s per warm frequency on the same machine, and bempp-cl's kernels are not
declared cacheable, so that cost is paid per process every time. A worker has to
solve about 40 frequencies to pay for its own start-up; below that a single warm
process wins by a wide margin.

Parallel sweeps support `progress_callback`, which stays in the parent process
and is driven by a queue the workers publish to, so progress arrives as
frequencies land rather than in order. `on_frequency_result` is rejected in
parallel mode: it exists to abort a sweep, and a worker cannot cancel
frequencies already running in its siblings.

## Half and Quarter Symmetry

Set `native_symmetry_plane` when the input mesh is a positive-side
mirror-reduced domain:

- `"yz"`: half model in `X >= 0`
- `"xz"`: half model in `Y >= 0`
- `"yz+xz"`: quarter model in `X >= 0, Y >= 0`

The solver mirrors the mesh as integration geometry, tests the BIE on only the
real half/quarter, and ties pressure coefficients across mirror orbits. This
preserves Bempp's singular quadrature on cut-plane seams while avoiding the
unused mirrored equation rows. Far-field pressure is evaluated from the
reconstructed full trace.

```python
config = SolveConfig(
    native_symmetry_plane="yz+xz",
    formulation=BIEFormulation.COMPLEX_K,
)
result = solve("waveguide-quarter.msh", config)
```

Native symmetry currently supports `STANDARD` and `COMPLEX_K` rigid Neumann
models. Burton-Miller, Robin `impedance_sources`, legacy `"xy"` symmetry, and
coupled infinite-baffle models fail explicitly rather than solving an
incorrect reduced open shell. When `gmres_restart` is left at `None`, the
symmetry path selects a restart length of 100 for robust half-model
convergence.

Reduced/full expanded parity validates the mirror reduction itself. It does
not remove the interior-resonance sensitivity of the underlying `STANDARD`
integral equation; the current small `COMPLEX_K` shift can also remain
mesh-sensitive near a resonance. Use independently validated frequency points
for production conclusions until a symmetry-compatible combined-field or
Burton-Miller path is available.

`ObservationConfig` builds polar observation arcs by default:

```python
from hornlab_bempp_bem import ObservationConfig, SolveConfig

config = SolveConfig(
    observation=ObservationConfig(
        planes=["horizontal", "vertical"],
        distance_m=2.0,
        angle_min_deg=0.0,
        angle_max_deg=180.0,
        angle_count=37,
        origin="mouth",
        sphere_grid=(37, 72),
    )
)
```

Allowed plane names are `"horizontal"`, `"vertical"`, and `"diagonal"`.

For exact observation coordinates, set `custom_points` to a mapping of plane
name to an `(N, 3)` array in metres. All requested planes must be present and
must have the same point count.

`sphere_grid=(n_theta, n_phi)` additionally evaluates a frame-relative
spherical field from the same solved system. Theta runs from the forward axis
through `sphere_theta_max_deg` (180° by default); phi wraps around the axis
without a duplicate 360° column. This field is independent of the selected
display planes and is suitable for solid-angle directivity integration.

## Formulations

`BIEFormulation.STANDARD` is the default direct Helmholtz boundary integral
solve.

`BIEFormulation.COMPLEX_K` applies a small complex wavenumber shift controlled
by `complex_k_shift`. This can reduce sensitivity to interior resonances while
keeping the standard operator structure.

`BIEFormulation.BURTON_MILLER` assembles the Burton-Miller combined equation
with hypersingular and adjoint double-layer operators. It is available for the
standard rigid/Neumann path. Robin wall admittance is intentionally not
implemented with Burton-Miller in this package.

## Robin Boundary Conditions

Set `impedance_sources` to map physical tag to normalized surface admittance:

```python
from hornlab_bempp_bem import SolveConfig

config = SolveConfig(
    impedance_sources={1: 0.05 + 0.0j},
)
```

The value is `beta = rho*c / Z_s`. `beta = 0` is rigid, and `beta = 1` is an
air-matched absorber. When non-empty, the solver substitutes the Robin
condition directly into the linear system and solves once. This direct path is
always double precision for numerical robustness, even when the requested
`precision` is `"single"`; each solver-log entry records both values. An AUTO
backend request falls back to Numba when OpenCL's selected devices lack fp64.
An explicit OpenCL request refuses the solve early and names the incapable
device, because silently undoing the required promotion would restore the
mixed-precision correctness defect.

## Assembly Backends

`assembly_backend="opencl"` uses Bempp's OpenCL backend. Install it with
`python -m pip install "hornlab-bempp-bem[opencl]"`; it requires both PyOpenCL
and a working OpenCL runtime.

`assembly_backend="numba"` uses Bempp's Numba backend. It needs a writable
Numba cache location; set `NUMBA_CACHE_DIR` when running in restricted
environments. The base package is sufficient for this path and does not install
PyOpenCL.

`assembly_backend="auto"` probes the requested OpenCL device, uses it when it
can be initialized, and otherwise falls back to the portable Numba backend.
Naming `"opencl"` explicitly keeps fail-fast behavior when OpenCL is required.
Automatic fallback emits one warning per sweep, and every solver-log entry
records the requested and effective backend, whether fallback occurred, and
the reason.

### What the Numba fallback costs

Falling back is not cheap, and on a stock install it is quiet. Measured on a
12-vCPU AVX2 Windows box against a three-rung ASRO68 ladder at 2 kHz, warm
assembly per frequency (`runs/260901-windows-b0/` carries the raw JSON and the
benchmark script):

Double precision:

| P1 DOF | Numba   | OpenCL  | ratio |
| ------ | ------- | ------- | ----- |
| 1,338  | 2.21 s  | 0.58 s  | 3.8x  |
| 2,912  | 15.36 s | 2.15 s  | 7.1x  |
| 4,992  | 67.90 s | 5.85 s  | 11.6x |

Single precision:

| P1 DOF | Numba   | OpenCL  | ratio |
| ------ | ------- | ------- | ----- |
| 1,338  | 2.40 s  | 0.38 s  | 6.3x  |
| 2,912  | 12.27 s | 1.23 s  | 10.0x |
| 4,992  | 58.07 s | 3.08 s  | 18.9x |

The gap widens with mesh size and is wider in single precision, because OpenCL
compiles a `vec8` kernel for fp32 against `vec4` for fp64 while Numba gains far
less from the narrower type. Post-solve field evaluation shows the same
asymmetry more sharply still: 0.416 s against 0.009 s at 4,992 DOF.

Two figures are worth keeping in view when reading those ratios. The linear
solve is unaffected — 8.05 s (Numba) against 8.36 s (OpenCL) at 4,992 DOF,
because it is the same SciPy LU either way — so the ratios describe assembly,
not end-to-end solve time. And OpenCL wins while using *fewer* cores, not more:
CPU-time over wall-time was 9.4 for Numba against 5.1 for OpenCL at 4,992 DOF.
That is a per-core efficiency difference, so throwing threads at the Numba path
does not close it.

### Checking the OpenCL runtime before you rely on it

`assembly_backend="auto"` already probes, and `"opencl"` already fails fast, so
the solve path reports a missing device. Post-solve field evaluation is the
deliberate exception — it falls back so that re-evaluating retained traces keeps
working on a machine whose solve ran elsewhere — and it now logs a warning when
it does, once per distinct reason.

To check before committing to a long sweep:

```python
import hornlab_bempp_bem as bempp_bem

check = bempp_bem.check_opencl("cpu")
print(check.describe())        # -> "OpenCL OK: assembled on 'AMD Ryzen 7 ...'"

bempp_bem.require_opencl()     # same thing, but raises OpenCLError
```

`check_opencl` assembles a real operator on a 32-element sphere rather than
enumerating devices. That is deliberate: neither `pyopencl.get_platforms()` nor
reading back `bempp_cl.api.DEFAULT_DEVICE_INTERFACE` survives a runtime that is
present and enumerable but cannot compile — which is exactly the unquoted
include-path failure described under [Performance](#performance). `check.stage`
reports how far it got (`pyopencl`, `device`, `assembly`, `ok`). It costs a
one-off kernel compile that the first real assembly would have paid anyway.

#### Getting a CPU OpenCL runtime

bempp-cl's OpenCL path needs a device of type **CPU**, not GPU. On Windows the
reference box uses the Intel CPU Runtime for OpenCL Applications, which despite
the name runs on AMD parts — the measurements above were taken on a Ryzen. It is
also on PyPI as `intel-opencl-rt`, which ships `win_amd64` wheels, so
`pip install intel-opencl-rt` is the quickest route. That package is
**proprietary** (Intel End User License Agreement for Developer Tools), so
installing it per-machine and redistributing it inside an installer are
different questions; the second is not settled here. It is deliberately *not*
declared as a dependency of this package.

On Linux, pocl is the usual permissively-licensed option. macOS exposes only a
GPU device, so there is no CPU OpenCL path there at all and bempp-cl takes the
Numba branch unconditionally.

## Outputs

`solve()` and `solve_frequencies()` return `SolveResult`.

Key result fields:

- `frequencies_hz`: `(F,)` solved frequencies in Hz
- `pressure_complex`: `(F, P, N)` complex pressure at observation points
- `spl_db`: `(F, P, N)` directivity normalized so the on-axis angle is `0 dB`
- `directivity_db`: hornlab-metal-bem-compatible alias for `spl_db`
- `impedance`: `(F,)` area-weighted average complex pressure on the source tag,
  in pascals per unit drive convention and not normalized to `rho*c`
- `observation_angles_deg`: `(N,)` polar angles in degrees
- `observation_points`: `(P, N, 3)` observation coordinates in metres
- `observation_planes`: plane names matching axis `P`
- `surface_pressure_avg`: source-tag keyed average surface pressure arrays
- `surface_pressure_complex`: optional `(F, n_p1_dofs)` P1 pressure
  coefficients when `return_surface_traces=True`
- `surface_neumann_complex`: optional `(F, n_dp0_dofs)` total `dp/dn`, including
  the Robin contribution, when `return_surface_traces=True`
- `sphere_pressure_complex`: optional `(F, n_theta*n_phi)` spherical pressure
- `sphere_theta_deg`, `sphere_phi_deg`: optional theta-major sphere coordinates
- `timings` and `solver_log`: timing plus requested/effective solver, precision,
  and assembly-backend metadata for each frequency

`spl_db` and `directivity_db` are not absolute SPL. Use `pressure_complex` for
absolute complex pressure and derive SPL explicitly when needed.

`evaluate_exterior_from_traces()` applies the same `DLP(p) - SLP(q)`
representation formula as solve-time observation evaluation. Trace arrays are
in Bempp P1/DP0 coefficient order and must be paired with the same mesh
connectivity and ordering. Complex pressure and both traces use the
`e^{-i omega t}` convention, the same convention as hornlab-metal-bem.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[dev]"  # includes OpenCL test dependencies
```

If you already have a HornLab development environment with `bempp_cl`, install
this repository editable into that interpreter instead.

## Tests

```bash
NUMBA_CACHE_DIR="$PWD/.numba_cache" python -m pytest -q
```

Slow validation tests that depend on external fixtures are expected to skip
when those fixtures are unavailable. To enable them without relying on a
particular workspace layout, set `HORNLAB_ASRO68_MESH` to the reference mesh,
and `HORNLAB_VALIDATION_ARTIFACTS` to the validation-artifact directory. The
ASRO68 gates compare against pinned library results and the independent ABEC
fixture; they do not depend on Waveguide Generator's deleted legacy solver.

### Smoke test

Upstream bempp-cl runs no Windows or macOS CI, so installing successfully is not
evidence that this machine can solve. `scripts/windows_smoke.py` establishes that
end to end — import, device selection, kernel build, assembly, LU solve, field
evaluation — and additionally asserts that OpenCL and Numba agree, which is the
only way to catch a runtime that is present and enumerable but wrong.

```bash
python scripts/windows_smoke.py
```

It exits non-zero on failure, so it can be wired into CI directly. `--json PATH`
writes a machine-readable report; `--refinement N` changes the sphere size. The
OpenCL arm is skipped, not failed, when there is no usable device — the missing
device is already reported as its own failed check. On the reference Windows box
the two backends agree to a relative difference of 4.5e-14 at `--refinement 2`.

Note that a deep working directory can fail the suite for an unrelated reason:
Numba's on-disk cache path can cross Windows' 260-character `MAX_PATH`, which
surfaces as a `FileNotFoundError` inside `numba.core.caching` rather than as
anything resembling a path-length problem. Set `NUMBA_CACHE_DIR` to a short path
if that happens.
