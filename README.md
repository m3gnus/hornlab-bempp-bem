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
`impedance_sources`, and serial or parallel frequency sweeps.

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

Signed-volume winding validation is applied to closed two-manifold meshes. It
is intentionally not used to flip or reject open surfaces because their signed
volume changes under a rigid translation; callers remain responsible for the
declared outward winding on open/bare meshes.

With `require_closed_mesh=True` (or `load_mesh(..., require_closed=True)`),
the surface must additionally be a closed 2-manifold: every edge shared by
exactly two triangles. Open edges and non-manifold edges are rejected. The
check also applies to pre-loaded `LoadedMesh` inputs passed to `solve()`.

Use `solve_frequencies(mesh, frequencies_hz, config=None)` when frequency
order comes from the caller instead of a generated sweep.

## Configuration

`SolveConfig` controls the solve.

Common fields:

- `freq_min_hz`, `freq_max_hz`, `freq_count`, `freq_spacing`
- `velocity_sources`, mapping physical tag to source weight
- `velocity_mode`, either `VelocityMode.ACCELERATION` or `VelocityMode.VELOCITY`
- `formulation`, one of `STANDARD`, `COMPLEX_K`, or `BURTON_MILLER`
- `solver`, one of `AUTO`, `LU`, or `GMRES`
- `observation`, an `ObservationConfig`
- `mesh_scale`
- `air_density`
- `require_closed_mesh`, reject open or non-manifold surfaces before solving
- `assembly_backend`, one of `"opencl"`, `"numba"`, or `"auto"`
- `opencl_device`, either `"cpu"` or `"gpu"` when using OpenCL
- `progress_callback`
- `on_frequency_result`, for streaming progress and early stop

`on_frequency_result` stops only when it returns exactly `False`; callbacks
used only for side effects may return `None` and the sweep continues. Serial and
parallel sweeps populate the same result fields, including
`surface_pressure_avg`.

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
    )
)
```

Allowed plane names are `"horizontal"`, `"vertical"`, and `"diagonal"`.

For exact observation coordinates, set `custom_points` to a mapping of plane
name to an `(N, 3)` array in metres. All requested planes must be present and
must have the same point count.

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
condition directly into the linear system and solves once.

## Assembly Backends

`assembly_backend="opencl"` uses Bempp's OpenCL backend. Install it with
`python -m pip install "hornlab-bempp-bem[opencl]"`; it requires both PyOpenCL
and a working OpenCL runtime.

`assembly_backend="numba"` uses Bempp's Numba backend. It needs a writable
Numba cache location; set `NUMBA_CACHE_DIR` when running in restricted
environments. The base package is sufficient for this path and does not install
PyOpenCL.

`assembly_backend="auto"` currently selects the production OpenCL Bempp backend.

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
- `timings` and `solver_log`: backend timing and diagnostic metadata

`spl_db` and `directivity_db` are not absolute SPL. Use `pressure_complex` for
absolute complex pressure and derive SPL explicitly when needed.

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
when those fixtures are unavailable.
