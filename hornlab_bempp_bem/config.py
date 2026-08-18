from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING, Callable, Literal

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


class BIEFormulation(Enum):
    STANDARD = "standard"
    BURTON_MILLER = "burton_miller"
    COMPLEX_K = "complex_k"


class LinearSolver(Enum):
    AUTO = "auto"
    LU = "lu"
    GMRES = "gmres"


class VelocityMode(Enum):
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"


class SourceMotion:
    """How a driven source tag's prescribed velocity maps onto its faces.

    NORMAL (default): each source face vibrates along its own outward normal at
    the same speed -- a uniformly breathing cap. AXIAL: the source moves as a
    rigid piston along its axis, so the per-face normal velocity is
    ``U * (n_hat . axis)`` (full at the pole, tapering to the rim) -- the
    realistic wavefront for a rigid dome/cone/diaphragm. A flat disc reduces
    exactly to NORMAL. Mirrors hornlab-metal-bem's SourceMotion for parity.
    """

    NORMAL = "normal"
    AXIAL = "axial"


AssemblyBackend = Literal["opencl", "numba", "auto"]
NativeSymmetryPlane = Literal["yz", "xz", "xy", "yz+xz"]
VectorizationMode = Literal["auto", "novec", "vec4", "vec8", "vec16"]
VECTORIZATION_MODES = ("auto", "novec", "vec4", "vec8", "vec16")
_MAX_SPHERE_POINTS = 100_000


def _is_integral_value(value: object) -> bool:
    """Return whether ``value`` represents an integer, excluding booleans."""
    if isinstance(value, bool):
        return False
    try:
        return isfinite(float(value)) and int(value) == value
    except (TypeError, ValueError, OverflowError):
        return False


@dataclass
class ObservationConfig:
    planes: list[str] = field(default_factory=lambda: ["horizontal", "vertical"])
    distance_m: float = 2.0
    angle_min_deg: float = 0.0
    angle_max_deg: float = 180.0
    angle_count: int = 37
    origin: Literal["mouth", "throat"] = "mouth"

    # Custom observation points: plane_name -> (N, 3) array.
    # When set, build_observation_points() returns these directly
    # instead of constructing polar arcs from the frame.
    custom_points: dict[str, NDArray[np.float64]] | None = None

    # Frame-relative spherical field grid as (n_theta, n_phi). Theta spans
    # [0, sphere_theta_max_deg] inclusive and phi spans [0, 360) without a
    # duplicate wrap column. The result is theta-major and uses the same
    # observation frame, origin, and distance as the requested polar cuts.
    sphere_grid: tuple[int, int] | None = None
    sphere_theta_max_deg: float = 180.0

    def __post_init__(self) -> None:
        try:
            distance_m = float(self.distance_m)
        except (TypeError, ValueError, OverflowError):
            distance_m = float("nan")
        if not isfinite(distance_m) or distance_m <= 0.0:
            raise ValueError("distance_m must be finite and greater than zero")
        self.distance_m = distance_m

        for field_name in ("angle_min_deg", "angle_max_deg"):
            try:
                value = float(getattr(self, field_name))
            except (TypeError, ValueError, OverflowError):
                value = float("nan")
            if not isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            setattr(self, field_name, value)

        if not _is_integral_value(self.angle_count) or self.angle_count < 1:
            raise ValueError("angle_count must be a positive integer")
        self.angle_count = int(self.angle_count)

        if self.origin not in ("mouth", "throat"):
            raise ValueError("origin must be 'mouth' or 'throat'")

        if self.sphere_grid is not None:
            try:
                grid = tuple(self.sphere_grid)
            except TypeError:
                raise ValueError(
                    "sphere_grid must be an iterable (n_theta, n_phi)"
                ) from None
            if len(grid) != 2:
                raise ValueError("sphere_grid must be (n_theta, n_phi)")
            if not all(_is_integral_value(value) for value in grid):
                raise ValueError("sphere_grid counts must be integers")
            n_theta, n_phi = int(grid[0]), int(grid[1])
            if n_theta < 2:
                raise ValueError("sphere_grid n_theta must be at least 2")
            if n_phi < 3:
                raise ValueError("sphere_grid n_phi must be at least 3")
            if n_theta * n_phi > _MAX_SPHERE_POINTS:
                raise ValueError(
                    "sphere_grid is too dense "
                    f"(n_theta*n_phi > {_MAX_SPHERE_POINTS})"
                )
            self.sphere_grid = (n_theta, n_phi)

        if not isinstance(self.planes, list):
            raise ValueError("planes must be a list of plane names")
        if not all(isinstance(plane, str) and plane for plane in self.planes):
            raise ValueError("planes must be a list of non-empty plane names")
        if len(set(self.planes)) != len(self.planes):
            raise ValueError("planes must not contain duplicates")
        if not self.planes and self.sphere_grid is None:
            raise ValueError("planes may be empty only when sphere_grid is set")
        if self.custom_points is None:
            unknown_planes = set(self.planes) - {
                "horizontal", "vertical", "diagonal",
            }
            if unknown_planes:
                names = ", ".join(sorted(repr(name) for name in unknown_planes))
                raise ValueError(f"planes contain unknown name(s): {names}")

        try:
            sphere_theta_max_deg = float(self.sphere_theta_max_deg)
        except (TypeError, ValueError, OverflowError):
            sphere_theta_max_deg = float("nan")
        if not (0.0 < sphere_theta_max_deg <= 180.0):
            raise ValueError("sphere_theta_max_deg must be in (0, 180]")
        self.sphere_theta_max_deg = sphere_theta_max_deg


@dataclass
class SolveConfig:

    # Frequency sweep
    freq_min_hz: float = 500.0
    freq_max_hz: float = 20_000.0
    freq_count: int = 40
    freq_spacing: Literal["log", "linear"] = "log"

    # BIE
    formulation: BIEFormulation = BIEFormulation.STANDARD
    complex_k_shift: float = 0.005
    slp_dlp_quadrature: int = 4
    slp_dlp_singular_quadrature: int = 4
    hyp_adlp_quadrature: int = 4

    # Linear solver
    solver: LinearSolver = LinearSolver.GMRES
    lu_threshold: int = 6000
    gmres_tol: float = 1e-6
    gmres_max_iter: int = 5000
    gmres_restart: int | None = None

    # Boundary condition
    velocity_mode: VelocityMode = VelocityMode.ACCELERATION
    velocity_sources: dict[int, float] = field(
        default_factory=lambda: {2: 1.0}
    )
    velocity_profile: Literal["piston", "dome", "ring"] = "piston"
    # Direction the prescribed source velocity acts in. "normal" (default) drives
    # each source face along its own outward normal (breathing cap); "axial"
    # drives the source as a rigid piston along its axis (v_n = U*(n_hat.axis)).
    # See SourceMotion. Default "normal" leaves every existing solve unchanged.
    source_motion: str = SourceMotion.NORMAL

    # Robin / impedance boundary condition (wall damping).
    # Maps physical tag -> normalized surface admittance β = ρc / Z_s
    # (dimensionless). β = 0 is rigid wall, β = 1 is air-matched absorber.
    # Light damping (the "clean up unrealistic dips" trick): β ~ 0.02-0.1.
    # When non-empty, the solver substitutes ∂p/∂n = i·k·β·p directly into
    # the BIE and solves once (no iteration); see _assemble_and_solve_impedance.
    impedance_sources: dict[int, complex] = field(default_factory=dict)

    # Observation
    observation: ObservationConfig = field(default_factory=ObservationConfig)

    # Frame override: skip infer_frame() when set.
    # Use this for enclosed geometries where the heuristic may get the
    # axis wrong, or when the caller has a known frame (e.g. WG bridge).
    frame_override: object | None = None  # ObservationFrame, kept as object to avoid circular import

    # Performance
    workers: int = 1
    precision: Literal["single", "double"] = "single"
    assembly_backend: AssemblyBackend = "opencl"
    opencl_device: Literal["cpu", "gpu"] = "cpu"
    # OpenCL kernel vector width. "auto" asks bempp-cl for the device's native
    # width, which is what it does by default.
    #
    # Forcing "vec16" above the native width is worth 1.07-1.09x on assemblies
    # with a large *trial* space -- measured on the 1209x4552-dof ASRO2 symmetry
    # path -- and nothing at all on small ones, where it is within noise or
    # marginally slower. The win depends on the trial count, not the test count,
    # which is why two independent benchmarks of the same idea disagreed until
    # they were compared on the same shape. "novec" is 3-5x slower and "vec4"
    # about 40% slower on this CPU; neither is ever worth selecting.
    #
    # Left at "auto" because the useful value depends on the device's native
    # width, and forcing 16 on hardware that is natively 16 (or 4) has not been
    # measured. Set it explicitly if you have benchmarked your own host.
    vectorization_mode: VectorizationMode = "auto"
    native_symmetry_plane: NativeSymmetryPlane | None = None
    restrict_neumann_space: bool = True

    # Retain the complete P1 pressure and total DP0 Neumann traces needed to
    # re-evaluate the exterior field after the solve. Robin faces include the
    # pressure-dependent contribution enforced by the assembled system.
    return_surface_traces: bool = False

    # Mesh scale (applied on load if mesh isn't already in metres)
    mesh_scale: float = 1.0

    # Require the loaded mesh to be a closed surface (no open boundary
    # edges). Set by callers solving closed-mode geometries (enclosure /
    # capped free-standing box). For a native-symmetry mesh, validation occurs
    # after mirroring so cut-plane edges are permitted but any other leak is
    # rejected. Also checked for pre-loaded LoadedMesh inputs. On an
    # intentionally open mesh, Bempp's default P1 pressure space constrains
    # free-boundary DOFs to zero; hornlab-metal-bem's all-vertex P1 space does
    # not, so open-mesh results are not directly comparable across backends.
    # Closed meshes use identical P1 spaces.
    require_closed_mesh: bool = False

    # Air density (kg/m^3). Default 1.2041 matches standard air at 20 C.
    air_density: float = 1.2041

    # Progress callback: called after each frequency solve.
    # Signature: (freq_index: int, total_freqs: int, frequency_hz: float) -> None
    progress_callback: Callable[[int, int, float], None] | None = None

    # Per-frequency result callback for early stopping.
    # Signature: (freq_index: int, frequency_hz: float, log_entry: dict) -> bool
    # Return exactly False to abort the sweep (partial SolveResult is built);
    # any other return value, including None, continues.
    on_frequency_result: Callable[[int, float, dict], bool] | None = None

    def __post_init__(self) -> None:
        if self.freq_spacing not in {"log", "linear"}:
            raise ValueError("freq_spacing must be 'log' or 'linear'")
        if not _is_integral_value(self.freq_count) or self.freq_count < 1:
            raise ValueError("freq_count must be at least 1")
        self.freq_count = int(self.freq_count)
        for field_name in ("freq_min_hz", "freq_max_hz"):
            value = getattr(self, field_name)
            try:
                valid = isfinite(value) and value > 0.0
            except (TypeError, ValueError):
                valid = False
            if not valid:
                raise ValueError(
                    f"{field_name} must be finite and greater than zero"
                )
        if self.freq_min_hz > self.freq_max_hz:
            raise ValueError("freq_min_hz must not exceed freq_max_hz")
        try:
            self.formulation = BIEFormulation(self.formulation)
        except (TypeError, ValueError):
            raise ValueError(
                "formulation must be 'standard', 'complex_k', or 'burton_miller'"
            ) from None
        try:
            self.solver = LinearSolver(self.solver)
        except (TypeError, ValueError):
            raise ValueError("solver must be 'auto', 'lu', or 'gmres'") from None
        try:
            valid_gmres_tol = isfinite(self.gmres_tol) and self.gmres_tol > 0.0
        except (TypeError, ValueError):
            valid_gmres_tol = False
        if not valid_gmres_tol:
            raise ValueError("gmres_tol must be finite and greater than zero")
        if self.gmres_restart is not None:
            if (
                not _is_integral_value(self.gmres_restart)
                or self.gmres_restart < 1
            ):
                raise ValueError(
                    "gmres_restart must be a positive integer or None"
                )
            self.gmres_restart = int(self.gmres_restart)
        try:
            self.velocity_mode = VelocityMode(self.velocity_mode)
        except (TypeError, ValueError):
            raise ValueError(
                "velocity_mode must be 'velocity' or 'acceleration'"
            ) from None
        if self.precision not in {"single", "double"}:
            raise ValueError("precision must be 'single' or 'double'")
        if self.assembly_backend not in {"opencl", "numba", "auto"}:
            raise ValueError(
                "assembly_backend must be one of: auto, opencl, numba"
            )
        if self.vectorization_mode not in VECTORIZATION_MODES:
            raise ValueError(
                "vectorization_mode must be one of: "
                + ", ".join(VECTORIZATION_MODES)
            )
        if not isinstance(self.restrict_neumann_space, bool):
            raise ValueError("restrict_neumann_space must be a boolean")
        if not isinstance(self.return_surface_traces, bool):
            raise ValueError("return_surface_traces must be a boolean")
        if self.native_symmetry_plane not in {None, "yz", "xz", "xy", "yz+xz"}:
            raise ValueError(
                "native_symmetry_plane must be None, 'yz', 'xz', 'xy', or 'yz+xz'"
            )
        if self.source_motion not in {SourceMotion.NORMAL, SourceMotion.AXIAL}:
            raise ValueError("source_motion must be 'normal' or 'axial'")
        if self.velocity_profile != "piston":
            raise NotImplementedError(
                "hornlab-bempp-bem only supports velocity_profile='piston'; "
                "use hornlab-metal-bem source_velocity_profiles for shaped "
                "source profiles"
            )


def reject_unsupported_native_symmetry(config: SolveConfig) -> None:
    """Validate symmetry modes implemented by the Bempp image assembler.

    Every unsupported symmetry combination is rejected here, before the mesh is
    loaded and the symmetry context is built, so a caller learns immediately
    rather than after the first frequency's assembly.
    """
    if config.native_symmetry_plane not in {None, "yz", "xz", "yz+xz"}:
        raise NotImplementedError(
            "hornlab-bempp-bem native symmetry supports transverse half/quarter "
            "models ('yz', 'xz', and 'yz+xz'); legacy 'xy' symmetry is not "
            "implemented"
        )
    if config.native_symmetry_plane is None:
        return
    # Mirrored by the same guards inside assemble_and_solve_symmetry, which
    # stay as a backstop for callers reaching that function directly.
    if config.formulation is BIEFormulation.BURTON_MILLER:
        raise NotImplementedError(
            "native half/quarter symmetry currently supports STANDARD and "
            "COMPLEX_K formulations; Burton-Miller is not implemented"
        )
    if config.impedance_sources:
        raise NotImplementedError(
            "native half/quarter symmetry with Robin impedance boundaries "
            "is not implemented yet"
        )
