"""In-process workarounds for bempp-cl's OpenCL program build path.

Two independent problems, both applied from ``device.configure_opencl``:

1. **Unquoted kernel include path.** bempp-cl passes its include directory to
   ``clBuildProgram`` unquoted, and pyopencl joins the option list on spaces, so
   any install under a path containing a space (``C:\\Users\\me\\My Projects``)
   splits the option and the build fails with ``INVALID_BUILD_OPTIONS``. pyopencl
   quotes its own include path, which is why only bempp-cl's is affected. The
   failure is also self-obscuring: pyopencl's cache error handler reads
   ``os.environ["PYOPENCL_CACHE_FAILURE_FATAL"]`` with no default, so the handler
   raises ``KeyError`` and destroys the real compiler diagnostic.

   These live here rather than only in the calling application because a
   ``spawn``-based parallel sweep starts fresh interpreters that inherit no
   monkeypatches: without this, every worker's OpenCL assembly fails.

2. **Redundant program rebuilds.**
   ``bempp_cl.core.opencl_kernels.get_kernel_from_operator_descriptor`` calls
   ``build_program`` unconditionally, and ``build_program`` re-enters the full
   pyopencl path each time::

       def build_program(assembly_function, options, precision, device_type="cpu"):
           kernel_string = open(kernel_file).read()
           kernel_options = get_kernel_compile_options(options, precision)
           return _cl.Program(default_context(device_type), kernel_string) \\
               .build(options=kernel_options).kernel_function

   pyopencl's *disk* cache avoids recompiling, but the call still pays a source
   read, a cache-key hash, a disk cache lookup, ``Program.build``, and the
   ``Program.__getattr__`` kernel-introspection path -- about 20-25 ms per call
   on the reference host. There are three such calls per operator assembly and
   four operator assemblies per frequency (boundary DLP and SLP, plus both
   far-field potentials), so a sweep pays this roughly twelve times per
   frequency.

   None of it depends on the frequency. The wavenumber reaches the kernel as a
   runtime buffer argument (``kernel_options_buffer`` in
   ``bempp_cl/core/opencl_assemblers.py``), never as a compile-time ``-D``
   macro: ``get_kernel_compile_options`` only emits the structural flags in
   ``options`` (``KERNEL_FUNCTION``, ``VEC_LENGTH``, ``VEC_STRING``,
   ``WORKGROUP_SIZE``, ``COMPLEX_KERNEL`` ...) plus the precision flag.
   Memoizing on the full build key leaves assembled matrices **bitwise
   identical** while cutting warm sweep time by 1.5-2.8x on production meshes.

Two hazards shape how the memo is built, both found in review:

* A built kernel belongs to the OpenCL **context** it was built against.
  bempp-cl's ``set_default_cpu_device`` builds a *new* context of the same type,
  after which a kernel cached under a context-free key is launched against
  buffers from the new context and fails with ``INVALID_MEM_OBJECT``. The
  context identity is therefore part of the key, along with the kernel and
  include paths.
* ``pyopencl.Kernel`` is **mutable**: invoking it sets its arguments and then
  enqueues. Without the memo every assembly built its own kernel object, so
  concurrent solves could not collide; sharing one would introduce that race.
  The cache is thread-local, which keeps each thread on its own kernel objects
  and preserves the full speed-up (bempp-cl's assemblers are serial within a
  thread, and the parallel sweep uses processes).
"""
from __future__ import annotations

import logging
import os
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)

_CACHE_FAILURE_FATAL_ENV = "PYOPENCL_CACHE_FAILURE_FATAL"

# Distinct programs needed by a whole sweep is in the low tens; this is a
# runaway guard for a long-lived server meeting many mesh shapes, not a tuning
# knob.
_MAX_CACHED_PROGRAMS = 256

_local = threading.local()
_generation = 0
_generation_lock = threading.Lock()

# Set by an explicit ``disable_opencl_program_cache()``. ``configure_opencl``
# enables the cache on every device setup, so without this a benchmark or test
# that turned it off would silently get it back on the next solve.
_opted_out = False


def enable_opencl_build_workarounds() -> bool:
    """Quote bempp-cl's kernel include path and give pyopencl its missing default.

    Idempotent and never raises; returns whether the include path needed
    quoting. Safe to call alongside an application-level shim doing the same
    thing -- an already-quoted path is left alone.
    """
    # Empty string is falsy, which is what pyopencl's handler expects when a
    # cache write fails but the build itself should still be reported.
    os.environ.setdefault(_CACHE_FAILURE_FATAL_ENV, "")

    try:
        from bempp_cl.core import opencl_kernels
    except Exception:  # pragma: no cover - depends on optional bempp-cl
        return False

    current = getattr(opencl_kernels, "_INCLUDE_PATH", None)
    if not isinstance(current, str) or not current:
        return False
    if current.startswith('"') and current.endswith('"'):
        return False
    if " " not in current:
        # Quoting would still be correct; leaving it alone keeps this a no-op
        # on the hosts that never needed it.
        return False

    opencl_kernels._INCLUDE_PATH = f'"{current}"'
    logger.debug("quoted bempp-cl kernel include path for clBuildProgram")
    return True


def _thread_cache() -> "OrderedDict[tuple, object]":
    """This thread's program cache, reset when the generation has moved on."""
    cache = getattr(_local, "programs", None)
    if cache is None or getattr(_local, "generation", None) != _generation:
        cache = OrderedDict()
        _local.programs = cache
        _local.generation = _generation
    return cache


def _context_identity(device_type: str) -> object:
    """Identify the OpenCL context a program would be built against.

    ``set_default_cpu_device`` replaces the context without changing the device
    *type*, so the type alone does not identify it. ``int_ptr`` is the handle
    pyopencl exposes for exactly this.
    """
    try:
        from bempp_cl.core.opencl_kernels import default_context

        return int(default_context(device_type).int_ptr)
    except Exception:  # pragma: no cover - runtime dependent
        # Unable to identify the context: fall back to a sentinel that is never
        # equal to a real handle, so nothing is served from cache.
        return object()


def _build_key(
    assembly_function: str,
    options: dict,
    precision: str,
    device_type: str,
) -> tuple:
    """Key covering every input ``build_program`` derives its result from.

    ``repr`` on the option values keeps a flag option (``{"X": None}``, emitted
    as a bare ``-D X``) distinct from the string ``"None"``.
    """
    from bempp_cl.core import opencl_kernels

    return (
        str(assembly_function),
        str(precision),
        str(device_type),
        _context_identity(device_type),
        str(getattr(opencl_kernels, "_KERNEL_PATH", "")),
        str(getattr(opencl_kernels, "_INCLUDE_PATH", "")),
        tuple(sorted((str(key), repr(value)) for key, value in options.items())),
    )


def _signature_is_compatible(original) -> bool:
    """Whether ``original`` can be called as bempp-cl's ``build_program``.

    Checking names alone is not enough: an upstream signature that made
    ``device_type`` keyword-only would pass a name check and then fail on the
    wrapper's positional call.
    """
    import inspect

    try:
        parameters = list(inspect.signature(original).parameters.values())
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    if len(parameters) != 4:
        return False
    positional = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    expected = ("assembly_function", "options", "precision", "device_type")
    return all(
        parameter.name == name and parameter.kind in positional
        for parameter, name in zip(parameters, expected)
    )


def enable_opencl_program_cache(*, force: bool = True) -> bool:
    """Memoize ``bempp_cl.core.opencl_kernels.build_program`` in process.

    Returns whether the cache is active. Safe to call repeatedly; a second call
    is a no-op. Returns ``False`` when bempp-cl is unavailable or its
    ``build_program`` signature no longer matches, leaving stock behaviour in
    place rather than risking a wrong kernel.

    ``force=False`` respects an earlier explicit ``disable_...`` call, which is
    what ``configure_opencl`` wants: it enables the cache on every device setup
    and must not undo a deliberate opt-out.
    """
    global _opted_out
    if not force and _opted_out:
        return False
    _opted_out = False

    try:
        import bempp_cl.core.opencl_kernels as opencl_kernels
    except Exception:  # pragma: no cover - depends on optional bempp-cl
        return False

    original = getattr(opencl_kernels, "build_program", None)
    if original is None:
        return False
    if getattr(original, "_hornlab_program_cache", False):
        return True
    if not _signature_is_compatible(original):
        logger.warning(
            "bempp-cl build_program signature changed; leaving the OpenCL "
            "program cache disabled",
        )
        return False

    def cached_build_program(
        assembly_function, options, precision, device_type="cpu",
    ):
        key = _build_key(assembly_function, options, precision, device_type)
        cache = _thread_cache()
        program = cache.get(key)
        if program is not None:
            cache.move_to_end(key)
            return program
        program = original(assembly_function, options, precision, device_type)
        cache[key] = program
        if len(cache) > _MAX_CACHED_PROGRAMS:
            cache.popitem(last=False)
        return program

    cached_build_program._hornlab_program_cache = True
    cached_build_program._hornlab_original = original
    opencl_kernels.build_program = cached_build_program
    logger.debug("bempp-cl OpenCL program cache enabled")
    return True


def disable_opencl_program_cache() -> bool:
    """Restore stock ``build_program`` and drop the cache. Mainly for tests.

    The opt-out sticks: a later ``configure_opencl`` will not silently switch
    the cache back on.
    """
    global _opted_out
    _opted_out = True

    try:
        import bempp_cl.core.opencl_kernels as opencl_kernels
    except Exception:  # pragma: no cover - depends on optional bempp-cl
        return False

    current = getattr(opencl_kernels, "build_program", None)
    original = getattr(current, "_hornlab_original", None)
    if original is None:
        return False
    opencl_kernels.build_program = original
    clear_opencl_program_cache()
    return True


def clear_opencl_program_cache() -> None:
    """Drop every memoized program, on every thread, without unpatching.

    Bumping a generation rather than clearing one dict is what makes this reach
    threads other than the caller's; each picks up a fresh cache on its next
    lookup.
    """
    global _generation
    with _generation_lock:
        _generation += 1
    _local.programs = OrderedDict()
    _local.generation = _generation


def opencl_program_cache_size() -> int:
    """Number of distinct OpenCL programs memoized for the calling thread."""
    return len(_thread_cache())
