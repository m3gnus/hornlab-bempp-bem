"""The OpenCL program cache against real bempp-cl, not a stub.

The stubbed unit tests define a program as a function of the four proposed key
inputs, so they cannot see a key that is missing something. These exercise the
real build path: the context a program is bound to, and whether assembled
matrices actually come out identical.
"""
from __future__ import annotations

import numpy as np
import pytest

from hornlab_bempp_bem._opencl_program_cache import (
    clear_opencl_program_cache,
    disable_opencl_program_cache,
    enable_opencl_program_cache,
    opencl_program_cache_size,
)

bempp_api = pytest.importorskip("bempp_cl.api")
pytest.importorskip("pyopencl")


@pytest.fixture
def opencl_cpu():
    from hornlab_bempp_bem.device import OpenCLError, configure_opencl

    try:
        configure_opencl("cpu")
    except OpenCLError as exc:
        pytest.skip(f"no usable OpenCL CPU device: {exc}")
    clear_opencl_program_cache()
    enable_opencl_program_cache()
    yield
    clear_opencl_program_cache()


def _assemble(space, k):
    kwargs = dict(assembler="dense", device_interface="opencl", precision="single")
    operator = bempp_api.operators.boundary.helmholtz.double_layer(
        space, space, space, k, **kwargs,
    )
    return np.asarray(bempp_api.as_matrix(operator.weak_form()))


@pytest.fixture
def sphere_space():
    grid = bempp_api.shapes.sphere(h=0.7)
    return bempp_api.function_space(grid, "P", 1)


def test_a_sweep_reuses_programs_and_stays_bitwise_identical(
    opencl_cpu, sphere_space,
):
    frequencies = [1.0, 1.5, 2.0]

    disable_opencl_program_cache()
    reference = [_assemble(sphere_space, k) for k in frequencies]

    clear_opencl_program_cache()
    enable_opencl_program_cache()
    cached = [_assemble(sphere_space, k) for k in frequencies]

    for expected, actual in zip(reference, cached):
        assert np.array_equal(expected, actual), "memoization changed the matrix"
    assert 0 < opencl_program_cache_size() <= 8, (
        "a whole sweep needs only a handful of distinct programs"
    )


def test_repeated_frequencies_do_not_grow_the_cache(opencl_cpu, sphere_space):
    _assemble(sphere_space, 1.0)
    after_first = opencl_program_cache_size()

    for k in (1.1, 1.2, 1.3, 1.4):
        _assemble(sphere_space, k)

    assert opencl_program_cache_size() == after_first, (
        "frequency must not be part of the build key"
    )


def test_a_new_context_is_not_served_a_stale_kernel(opencl_cpu, sphere_space):
    """``set_default_cpu_device`` builds a new context of the same type.

    A key without the context handle returns the old context's kernel, and the
    assembler then launches it against buffers from the new one, failing with
    ``INVALID_MEM_OBJECT``.
    """
    import bempp_cl.core.opencl_kernels as opencl_kernels

    _assemble(sphere_space, 1.0)
    before = int(opencl_kernels.default_context("cpu").int_ptr)

    try:
        opencl_kernels.set_default_cpu_device(0, 0)
    except Exception as exc:  # pragma: no cover - platform dependent
        pytest.skip(f"cannot rebind the default CPU device here: {exc}")

    after = int(opencl_kernels.default_context("cpu").int_ptr)
    if after == before:
        pytest.skip("the driver reused the same context; nothing to prove")

    # Must assemble cleanly against the new context rather than raising.
    matrix = _assemble(sphere_space, 1.0)

    assert np.isfinite(matrix).all()


def test_disable_restores_stock_behaviour(opencl_cpu, sphere_space):
    import bempp_cl.core.opencl_kernels as opencl_kernels

    assert getattr(opencl_kernels.build_program, "_hornlab_program_cache", False)

    assert disable_opencl_program_cache() is True

    assert not getattr(
        opencl_kernels.build_program, "_hornlab_program_cache", False,
    )
    assert np.isfinite(_assemble(sphere_space, 1.0)).all()
