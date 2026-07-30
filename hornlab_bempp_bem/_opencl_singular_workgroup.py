"""Pick the largest *exact* workgroup size for bempp-cl's singular assembler.

bempp-cl fixes ``WORKGROUP_SIZE_GALERKIN = 16`` and gives each singular
element-pair a workgroup of that many work items, splitting the pair's
quadrature points between them::

    # bempp_cl/core/opencl_assemblers.py
    local_quad_points = number_of_quad_points // WORKGROUP_SIZE_GALERKIN

    # sources/kernels/evaluate_dense_singular.cl
    for (uint quadIndex = localQuadPointsPerItem * localId;
         quadIndex < localQuadPointsPerItem * (localId + 1); ++quadIndex)

The division is a floor, so the kernel integrates only
``WORKGROUP_SIZE * (npoints // WORKGROUP_SIZE)`` points and **silently discards
the remainder**. That has two consequences.

**It is an upstream accuracy bug at some quadrature orders.** The singular point
counts per adjacency class, measured on a real mesh:

    order 2   32,   80,   96    divisible by 16
    order 3  162,  405,  486    NOT divisible by 16 -- 2, 5 and 6 points dropped
    order 4  512, 1280, 1536    divisible by 16, 32, 64 and 128
    order 5 1250, 3125, 3750    NOT divisible by 16 -- 2, 5 and 6 points dropped
    order 6 2592, 6480, 7776    divisible by 16

So stock bempp-cl quietly drops quadrature points at odd singular orders. It is
tempting to blame that for the accuracy penalty already recorded at singular
order 3, but **measurement says otherwise**: integrating order 3 exactly (with a
workgroup of 1, which divides all three counts) lands within `0.006 dB` of the
stock truncated result, both about `1.02 dB` from order 4 on the reference
quarter. The order-3 penalty is the order, not the missing points. The dropped
points remain a real upstream defect worth reporting -- they carry very little
quadrature weight here, but nothing guarantees that on another geometry.

**It leaves performance on the table at order 4**, this package's default and
its measured accuracy floor. Every count there divides 64, so a workgroup of 64
integrates exactly the same points in exactly the same way, just spread over
more work items: measured 1.21x faster on the whole double-layer assembly for
both the 458-triangle and 2275-triangle quarter meshes, with a maximum relative
matrix difference of 8e-8 -- floating-point reordering only.

This module therefore picks, per assembly, the largest size in ``(64, 32, 16)``
that divides every point count. When none does -- orders 3 and 5, where stock
bempp-cl is already dropping points -- it leaves the stock size alone and warns,
rather than silently changing which points are discarded or falling back to a
workgroup of one and becoming an order of magnitude slower.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Descending; 16 is bempp-cl's stock value and the floor we never go below.
_CANDIDATE_SIZES = (64, 32, 16)

_warned_counts: set[tuple[int, ...]] = set()


def safe_workgroup_size(number_of_quad_points) -> int | None:
    """Largest candidate size dividing every per-pair point count.

    ``None`` means even the stock 16 does not divide them, so bempp-cl is
    already discarding quadrature points and we must not make it worse.
    """
    counts = np.unique(np.asarray(number_of_quad_points).reshape(-1))
    if counts.size == 0:
        return _CANDIDATE_SIZES[-1]
    for size in _CANDIDATE_SIZES:
        if np.all(counts % size == 0):
            return int(size)
    return None


def _warn_once_about_dropped_points(number_of_quad_points) -> None:
    counts = tuple(
        int(value)
        for value in np.unique(np.asarray(number_of_quad_points).reshape(-1))
    )
    if counts in _warned_counts:
        return
    _warned_counts.add(counts)
    logger.warning(
        "bempp-cl's singular quadrature point counts %s are not divisible by "
        "its workgroup size 16, so it is silently discarding %s points per "
        "element pair. This is an upstream defect; results at this singular "
        "quadrature order are less accurate than the order implies. Prefer "
        "singular quadrature order 4.",
        list(counts), [int(c) % 16 for c in counts],
    )


def enable_singular_workgroup_tuning() -> bool:
    """Patch bempp-cl's singular assembler to size its workgroups exactly.

    Idempotent; returns whether the tuning is active. Returns ``False`` and
    leaves bempp-cl untouched when its signature is not the one this wraps.
    """
    try:
        import bempp_cl.core.opencl_assemblers as opencl_assemblers
    except Exception:  # pragma: no cover - depends on optional bempp-cl
        return False

    original = getattr(opencl_assemblers, "singular_assembler", None)
    if original is None:
        return False
    if getattr(original, "_hornlab_workgroup_tuned", False):
        return True

    import inspect

    try:
        parameters = list(inspect.signature(original).parameters)
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        return False
    if "number_of_quad_points" not in parameters:
        logger.warning(
            "bempp-cl singular_assembler signature changed (%s); leaving the "
            "workgroup size alone", parameters,
        )
        return False
    quad_index = parameters.index("number_of_quad_points")

    def tuned_singular_assembler(*args, **kwargs):
        if "number_of_quad_points" in kwargs:
            counts = kwargs["number_of_quad_points"]
        elif len(args) > quad_index:
            counts = args[quad_index]
        else:  # pragma: no cover - defensive
            return original(*args, **kwargs)

        size = safe_workgroup_size(counts)
        if size is None:
            _warn_once_about_dropped_points(counts)
            return original(*args, **kwargs)

        # The stock assembler reads this module global both for the kernel's
        # compile-time -D WORKGROUP_SIZE and for its local work size, so they
        # stay consistent. Mutating a global around the call is only safe
        # because bempp-cl's assemblers are serial within a process; the
        # parallel sweep uses processes, not threads.
        previous = opencl_assemblers.WORKGROUP_SIZE_GALERKIN
        opencl_assemblers.WORKGROUP_SIZE_GALERKIN = size
        try:
            return original(*args, **kwargs)
        finally:
            opencl_assemblers.WORKGROUP_SIZE_GALERKIN = previous

    tuned_singular_assembler._hornlab_workgroup_tuned = True
    tuned_singular_assembler._hornlab_original = original
    # The dispatcher imports this name inside the call, so rebinding the module
    # attribute is picked up without any stale reference to fix up.
    opencl_assemblers.singular_assembler = tuned_singular_assembler
    logger.debug("bempp-cl singular workgroup tuning enabled")
    return True


def disable_singular_workgroup_tuning() -> bool:
    """Restore bempp-cl's stock singular assembler. Mainly for tests."""
    try:
        import bempp_cl.core.opencl_assemblers as opencl_assemblers
    except Exception:  # pragma: no cover - depends on optional bempp-cl
        return False

    current = getattr(opencl_assemblers, "singular_assembler", None)
    original = getattr(current, "_hornlab_original", None)
    if original is None:
        return False
    opencl_assemblers.singular_assembler = original
    return True
