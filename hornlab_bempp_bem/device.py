from __future__ import annotations

from dataclasses import dataclass


class OpenCLError(RuntimeError):
    pass


# Which device type bempp's globals are currently pointing at, and the name we
# resolved for it. This used to be an ``lru_cache(maxsize=2)``, which made
# cpu -> gpu -> cpu skip the third call entirely: the cached CPU name came back
# while ``BOUNDARY_OPERATOR_DEVICE_TYPE`` was still ``"gpu"``. Remembering only
# the *current* type means a repeat call is still cheap but a switch always
# reapplies.
_active_device_type: str | None = None
_active_device_name: str | None = None


def reset_opencl_device() -> None:
    """Forget the configured device so the next call reconfigures from scratch."""
    global _active_device_type, _active_device_name
    _active_device_type = None
    _active_device_name = None


def set_vectorization_mode(mode: str = "auto") -> str:
    """Select the OpenCL kernel vector width bempp-cl compiles for.

    bempp-cl reads ``bempp_cl.api.VECTORIZATION_MODE`` when it builds kernel
    options, so the width lands in the program cache key and each mode gets its
    own cached program. ``"auto"`` is bempp-cl's own behaviour: the device's
    native width. See ``SolveConfig.vectorization_mode`` for when forcing a
    wider one is worth it.
    """
    normalized = str(mode or "auto").strip().lower()
    valid = {"auto", "novec", "vec4", "vec8", "vec16"}
    if normalized not in valid:
        raise OpenCLError(
            f"vectorization_mode must be one of {sorted(valid)}"
        )
    try:
        import bempp_cl.api as bempp_api
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise OpenCLError(f"bempp-cl is unavailable: {exc}") from exc
    bempp_api.VECTORIZATION_MODE = normalized
    return normalized


def configure_opencl(device_type: str = "cpu") -> str:
    """Configure bempp-cl to use a concrete OpenCL device type."""
    global _active_device_type, _active_device_name

    normalized = str(device_type or "cpu").strip().lower()
    if normalized not in {"cpu", "gpu"}:
        raise OpenCLError("opencl_device must be 'cpu' or 'gpu'")

    if normalized == _active_device_type and _active_device_name is not None:
        return _active_device_name

    try:
        import pyopencl  # noqa: F401
    except ModuleNotFoundError as exc:
        raise OpenCLError(
            "The OpenCL backend requires the optional PyOpenCL dependency. "
            "Install hornlab-bempp-bem[opencl], or use "
            "assembly_backend='numba'."
        ) from exc

    try:
        import bempp_cl.api as bempp_api
        import bempp_cl.core.opencl_kernels as opencl_kernels
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise OpenCLError(f"bempp-cl OpenCL runtime is unavailable: {exc}") from exc

    # Both are bempp-cl defects worked around in-process; see
    # _opencl_program_cache. The build workarounds must run before the first
    # kernel build (they fix an install path containing a space), and the
    # program cache is a pure memoization that leaves assembled matrices
    # bitwise identical. Applying them here rather than in the calling
    # application is what makes a spawn-based parallel sweep work: fresh
    # interpreters inherit no monkeypatches, and every worker reaches this
    # function before it assembles anything.
    from ._opencl_program_cache import (
        clear_opencl_program_cache,
        enable_opencl_build_workarounds,
        enable_opencl_program_cache,
    )
    from ._opencl_singular_workgroup import enable_singular_workgroup_tuning

    enable_opencl_build_workarounds()
    enable_opencl_program_cache(force=False)
    # bempp-cl hardcodes a 16-item workgroup for the singular assembler even
    # when the quadrature point counts allow a larger exact one. See
    # _opencl_singular_workgroup -- it also surfaces an upstream defect where
    # points are discarded at odd quadrature orders.
    enable_singular_workgroup_tuning()

    try:
        setattr(bempp_api, "BOUNDARY_OPERATOR_DEVICE_TYPE", normalized)
        setattr(bempp_api, "POTENTIAL_OPERATOR_DEVICE_TYPE", normalized)

        if normalized == "cpu":
            device = opencl_kernels.default_cpu_device()
            opencl_kernels.default_cpu_context()
        else:
            device = opencl_kernels.default_gpu_device()
            opencl_kernels.default_gpu_context()
            # bempp-cl dense singular assembly still needs a CPU context.
            opencl_kernels.default_cpu_device()
            opencl_kernels.default_cpu_context()
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise OpenCLError(
            f"OpenCL {normalized} device could not be initialized. "
            "Install a CPU OpenCL driver (e.g. pocl on Linux, the Intel CPU "
            "runtime on Windows) or use assembly_backend='numba'."
        ) from exc

    if _active_device_type is not None and _active_device_type != normalized:
        # Switching device type replaces the contexts programs were built
        # against. The cache key carries the context handle so a stale entry
        # could not be served anyway, but dropping them frees the old contexts.
        clear_opencl_program_cache()

    _active_device_type = normalized
    _active_device_name = str(
        getattr(device, "name", None) or f"OpenCL {normalized.upper()}"
    )
    return _active_device_name


def _opencl_device_supports_fp64(device: object) -> bool:
    """Return whether an OpenCL device advertises double-precision support."""
    try:
        if int(getattr(device, "double_fp_config", 0)) != 0:
            return True
    except (TypeError, ValueError):
        pass
    extensions = set(str(getattr(device, "extensions", "")).split())
    return bool({"cl_khr_fp64", "cl_amd_fp64"} & extensions)


def opencl_devices_without_fp64(device_type: str = "cpu") -> tuple[str, ...]:
    """Name configured devices that Bempp needs but that lack fp64 support.

    Bempp's GPU boundary path still uses its CPU OpenCL context for dense
    singular assembly, so both devices must support fp64 in GPU mode.
    """
    normalized = str(device_type or "cpu").strip().lower()
    configure_opencl(normalized)

    import bempp_cl.core.opencl_kernels as opencl_kernels

    if normalized == "cpu":
        devices = (opencl_kernels.default_cpu_device(),)
    else:
        devices = (
            opencl_kernels.default_gpu_device(),
            opencl_kernels.default_cpu_device(),
        )
    return tuple(
        str(getattr(device, "name", None) or "unnamed OpenCL device")
        for device in devices
        if not _opencl_device_supports_fp64(device)
    )


# Compatibility with the previous ``lru_cache``-decorated implementation, which
# callers and tests reset through this name.
configure_opencl.cache_clear = reset_opencl_device


@dataclass(frozen=True)
class OpenCLCheck:
    """Outcome of :func:`check_opencl`."""

    ok: bool
    stage: str
    device_name: str | None = None
    detail: str | None = None

    def describe(self) -> str:
        """One line suitable for a log, a banner, or a smoke test."""
        if self.ok:
            return f"OpenCL OK: assembled on {self.device_name!r}"
        return f"OpenCL UNAVAILABLE at stage {self.stage!r}: {self.detail}"


def check_opencl(device_type: str = "cpu") -> OpenCLCheck:
    """Verify the OpenCL path by *using* it, and report where it breaks.

    Two cheaper checks get proposed for this job and both are weaker:

    ``pyopencl.get_platforms()`` finding a CPU-type device proves only that an
    ICD is registered. It says nothing about whether bempp-cl chose it, and
    nothing about whether a kernel will build.

    Reading back ``bempp_cl.api.DEFAULT_DEVICE_INTERFACE`` after import is
    better -- it is the actual decision variable, set at import by a probe in
    ``bempp_cl/api/__init__.py`` that looks for a CPU-type device and falls
    back to ``"numba"`` when it finds none. But it is still only a record of
    that probe. It is skipped entirely on Darwin, it can be reassigned by any
    caller afterwards, and this package overrides it per-operator anyway by
    passing ``device_interface`` explicitly.

    Neither survives the failure that actually bites: a runtime that is present
    and enumerable but cannot compile. That is not hypothetical here -- see
    ``_opencl_program_cache.enable_opencl_build_workarounds``, which exists
    because an install path containing a space breaks the kernel build on a
    machine whose device enumerates perfectly.

    So this builds a real kernel and assembles a real operator on a 32-element
    sphere. It is the only version that fails when the runtime is broken rather
    than absent, and the cost is a one-off: it pays the same kernel compile the
    first real assembly would have paid anyway, so run it at startup and it is
    not extra work, it is the same work moved somewhere it can be reported.

    ``stage`` says how far it got: ``pyopencl`` (module missing), ``device``
    (no usable device), ``assembly`` (device present, kernel or assembly
    failed), or ``ok``.
    """
    try:
        import pyopencl  # noqa: F401
    except Exception as exc:
        return OpenCLCheck(False, "pyopencl", None, f"PyOpenCL is unavailable: {exc}")

    try:
        # Runtimes pad the device name; it ends up in logs and banners.
        device_name = configure_opencl(device_type).strip()
    except Exception as exc:
        return OpenCLCheck(False, "device", None, str(exc))

    try:
        import numpy as np
        import bempp_cl.api as bempp_api
        from bempp_cl.api.operators.boundary import helmholtz

        grid = bempp_api.shapes.regular_sphere(1)
        space = bempp_api.function_space(grid, "P", 1)
        # to_dense(), not the ``.A`` property: bempp-cl 0.4.2 deprecates ``.A``
        # and ``.A`` is only a warning wrapper around this same call.
        matrix = np.asarray(
            helmholtz.single_layer(
                space, space, space, 1.0,
                assembler="dense", device_interface="opencl",
            ).weak_form().to_dense()
        )
        if matrix.shape != (space.global_dof_count, space.global_dof_count):
            raise OpenCLError(f"unexpected operator shape {matrix.shape}")
        if not np.all(np.isfinite(matrix.view(np.float64))):
            raise OpenCLError("assembled operator contains non-finite entries")
    except Exception as exc:
        return OpenCLCheck(False, "assembly", device_name, f"{type(exc).__name__}: {exc}")

    return OpenCLCheck(True, "ok", device_name, None)


def require_opencl(device_type: str = "cpu") -> OpenCLCheck:
    """:func:`check_opencl`, but raise ``OpenCLError`` when it is not usable.

    For callers that would rather stop at startup than discover the problem as
    an unexplained several-fold slowdown partway through a sweep.
    """
    result = check_opencl(device_type)
    if not result.ok:
        raise OpenCLError(result.describe())
    return result
