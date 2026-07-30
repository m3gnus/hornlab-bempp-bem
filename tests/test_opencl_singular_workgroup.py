"""Singular workgroup sizing must be exact or leave bempp-cl alone.

bempp-cl's singular kernel integrates only
``WORKGROUP_SIZE * (npoints // WORKGROUP_SIZE)`` quadrature points and silently
discards the remainder, so the workgroup size may only be raised when it
divides every per-element-pair point count exactly.
"""
from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

from hornlab_bempp_bem._opencl_singular_workgroup import (
    _CANDIDATE_SIZES,
    _warned_counts,
    disable_singular_workgroup_tuning,
    enable_singular_workgroup_tuning,
    safe_workgroup_size,
)

# Measured on a real mesh: the distinct singular point counts bempp-cl produces
# at each quadrature order, per adjacency class.
POINT_COUNTS_BY_ORDER = {
    2: [32, 80, 96],
    3: [162, 405, 486],
    4: [512, 1280, 1536],
    5: [1250, 3125, 3750],
    6: [2592, 6480, 7776],
}


@pytest.mark.parametrize(
    ("order", "expected"),
    [(2, 16), (3, None), (4, 64), (5, None), (6, 16)],
)
def test_real_quadrature_orders_pick_the_right_size(order, expected):
    counts = np.array(POINT_COUNTS_BY_ORDER[order], dtype=np.int64)

    assert safe_workgroup_size(counts) == expected


def test_the_default_order_gets_the_largest_size():
    """Order 4 is this package's default and its measured accuracy floor."""
    assert safe_workgroup_size(
        np.array(POINT_COUNTS_BY_ORDER[4], dtype=np.int64)
    ) == max(_CANDIDATE_SIZES)


@pytest.mark.parametrize("order", [3, 5])
def test_orders_where_bempp_already_drops_points_are_refused(order):
    """Returning None keeps stock behaviour instead of changing what is lost."""
    counts = np.array(POINT_COUNTS_BY_ORDER[order], dtype=np.int64)

    assert safe_workgroup_size(counts) is None
    # ... and stock really is dropping points there.
    assert np.any(counts % 16 != 0)


def test_a_chosen_size_always_divides_every_count():
    for counts in POINT_COUNTS_BY_ORDER.values():
        array = np.array(counts, dtype=np.int64)
        size = safe_workgroup_size(array)
        if size is not None:
            assert np.all(array % size == 0)


def test_the_size_is_never_below_bempp_s_own():
    for counts in POINT_COUNTS_BY_ORDER.values():
        size = safe_workgroup_size(np.array(counts, dtype=np.int64))
        assert size is None or size >= min(_CANDIDATE_SIZES)


def test_a_single_odd_count_blocks_the_whole_assembly():
    """One bad pair is enough; the check is over every count, not the median."""
    assert safe_workgroup_size(np.array([512, 1280, 1537])) is None


def test_empty_counts_fall_back_to_the_stock_size():
    assert safe_workgroup_size(np.array([], dtype=np.int64)) == min(_CANDIDATE_SIZES)


# --------------------------------------------------------------------------
# Patching behaviour, against a stub so it runs without an OpenCL device.
# --------------------------------------------------------------------------

@pytest.fixture
def fake_bempp(monkeypatch):
    seen: list[int] = []

    def singular_assembler(
        device_interface, operator_descriptor, grid, domain, dual_to_range,
        test_points, trial_points, quad_weights, test_elements, trial_elements,
        test_offsets, trial_offsets, weights_offsets, number_of_quad_points,
        kernel_options, result,
    ):
        seen.append(module.WORKGROUP_SIZE_GALERKIN)

    module = types.ModuleType("bempp_cl.core.opencl_assemblers")
    module.singular_assembler = singular_assembler
    module.WORKGROUP_SIZE_GALERKIN = 16

    core = types.ModuleType("bempp_cl.core")
    core.opencl_assemblers = module
    root = types.ModuleType("bempp_cl")
    root.core = core
    monkeypatch.setitem(sys.modules, "bempp_cl", root)
    monkeypatch.setitem(sys.modules, "bempp_cl.core", core)
    monkeypatch.setitem(sys.modules, "bempp_cl.core.opencl_assemblers", module)
    _warned_counts.clear()
    yield module, seen
    disable_singular_workgroup_tuning()
    _warned_counts.clear()


def _call(module, counts):
    module.singular_assembler(
        "opencl", None, None, None, None, None, None, None, None, None,
        None, None, None, np.array(counts, dtype=np.int64), None, None,
    )


def test_the_assembler_sees_the_tuned_size(fake_bempp):
    module, seen = fake_bempp
    assert enable_singular_workgroup_tuning() is True

    _call(module, POINT_COUNTS_BY_ORDER[4])

    assert seen == [64]


def test_the_global_is_restored_after_each_assembly(fake_bempp):
    module, _seen = fake_bempp
    enable_singular_workgroup_tuning()

    _call(module, POINT_COUNTS_BY_ORDER[4])

    assert module.WORKGROUP_SIZE_GALERKIN == 16, "must not leak between calls"


def test_the_global_is_restored_even_when_the_assembly_raises(fake_bempp):
    module, _seen = fake_bempp

    def exploding(*args, **kwargs):
        raise RuntimeError("assembly failed")

    module.singular_assembler = exploding
    enable_singular_workgroup_tuning()

    with pytest.raises(RuntimeError):
        _call(module, POINT_COUNTS_BY_ORDER[4])

    assert module.WORKGROUP_SIZE_GALERKIN == 16


def test_an_unsafe_order_keeps_the_stock_size_and_warns(fake_bempp, caplog):
    module, seen = fake_bempp
    enable_singular_workgroup_tuning()

    with caplog.at_level(logging.WARNING, logger="hornlab_bempp_bem"):
        _call(module, POINT_COUNTS_BY_ORDER[3])

    assert seen == [16], "must not change which points get discarded"
    assert "silently discarding" in caplog.text


def test_the_dropped_point_warning_is_not_repeated(fake_bempp, caplog):
    module, _seen = fake_bempp
    enable_singular_workgroup_tuning()

    with caplog.at_level(logging.WARNING, logger="hornlab_bempp_bem"):
        for _ in range(5):
            _call(module, POINT_COUNTS_BY_ORDER[3])

    assert caplog.text.count("silently discarding") == 1


def test_enable_is_idempotent_and_disable_restores_stock(fake_bempp):
    module, _seen = fake_bempp
    original = module.singular_assembler

    assert enable_singular_workgroup_tuning() is True
    patched = module.singular_assembler
    assert enable_singular_workgroup_tuning() is True
    assert module.singular_assembler is patched

    assert disable_singular_workgroup_tuning() is True
    assert module.singular_assembler is original
    assert disable_singular_workgroup_tuning() is False


def test_an_unexpected_signature_leaves_bempp_untouched(fake_bempp):
    module, _seen = fake_bempp

    def renamed(device_interface, npoints, result):
        return None

    module.singular_assembler = renamed

    assert enable_singular_workgroup_tuning() is False
    assert module.singular_assembler is renamed


def test_missing_bempp_is_not_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "bempp_cl.core.opencl_assemblers", None)

    assert enable_singular_workgroup_tuning() is False
