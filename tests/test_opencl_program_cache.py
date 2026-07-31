"""The bempp-cl OpenCL program cache must be a pure, reversible memoization."""
from __future__ import annotations

import sys
import types

import pytest

from hornlab_bempp_bem._opencl_program_cache import (
    _build_key,
    clear_opencl_program_cache,
    disable_opencl_program_cache,
    enable_opencl_program_cache,
    opencl_program_cache_size,
)


class _FakeContext:
    def __init__(self, int_ptr):
        self.int_ptr = int_ptr


@pytest.fixture
def fake_bempp(monkeypatch):
    """Install a stub ``bempp_cl.core.opencl_kernels`` with a counting build."""
    calls: list[tuple] = []

    def build_program(assembly_function, options, precision, device_type="cpu"):
        calls.append((assembly_function, dict(options), precision, device_type))
        return f"program::{assembly_function}::{len(calls)}"

    module = types.ModuleType("bempp_cl.core.opencl_kernels")
    module.build_program = build_program
    module._KERNEL_PATH = r"C:\kernels"
    module._INCLUDE_PATH = r'"C:\kernels\include"'
    module._contexts = {"cpu": _FakeContext(0x1000), "gpu": _FakeContext(0x2000)}
    module.default_context = lambda device_type: module._contexts[device_type]

    core = types.ModuleType("bempp_cl.core")
    core.opencl_kernels = module
    root = types.ModuleType("bempp_cl")
    root.core = core

    monkeypatch.setitem(sys.modules, "bempp_cl", root)
    monkeypatch.setitem(sys.modules, "bempp_cl.core", core)
    monkeypatch.setitem(sys.modules, "bempp_cl.core.opencl_kernels", module)
    clear_opencl_program_cache()
    yield module, calls
    disable_opencl_program_cache()
    clear_opencl_program_cache()


def test_identical_arguments_build_once(fake_bempp):
    module, calls = fake_bempp
    assert enable_opencl_program_cache() is True

    options = {"KERNEL_FUNCTION": "helmholtz_double_layer", "VEC_LENGTH": 8}
    first = module.build_program("dense_regular_vec", options, "single", "cpu")
    second = module.build_program("dense_regular_vec", options, "single", "cpu")

    assert first == second
    assert len(calls) == 1, "the second identical build must be served from cache"


def test_frequency_sweep_reuses_one_program(fake_bempp):
    """A sweep varies only the runtime wavenumber, never the build key."""
    module, calls = fake_bempp
    enable_opencl_program_cache()

    for _frequency in (500.0, 750.0, 1000.0, 1500.0, 2000.0):
        module.build_program(
            "dense_regular_vec",
            {"KERNEL_FUNCTION": "helmholtz_single_layer", "VEC_LENGTH": 8},
            "single",
            "cpu",
        )

    assert len(calls) == 1
    assert opencl_program_cache_size() == 1


@pytest.mark.parametrize(
    "changed",
    [
        {"assembly_function": "dense_regular_novec"},
        {"precision": "double"},
        {"device_type": "gpu"},
        {"options": {"KERNEL_FUNCTION": "helmholtz_single_layer", "VEC_LENGTH": 8}},
        {"options": {"KERNEL_FUNCTION": "helmholtz_double_layer", "VEC_LENGTH": 4}},
        {"options": {"KERNEL_FUNCTION": "helmholtz_double_layer", "VEC_LENGTH": 8,
                     "COMPLEX_KERNEL": None}},
    ],
)
def test_every_build_input_separates_cache_entries(fake_bempp, changed):
    module, calls = fake_bempp
    enable_opencl_program_cache()

    base = dict(
        assembly_function="dense_regular_vec",
        options={"KERNEL_FUNCTION": "helmholtz_double_layer", "VEC_LENGTH": 8},
        precision="single",
        device_type="cpu",
    )
    variant = {**base, **changed}

    module.build_program(**base)
    module.build_program(**variant)

    assert len(calls) == 2, f"changing {sorted(changed)} must not hit the cache"


def test_flag_option_is_distinct_from_the_string_none():
    """``{"X": None}`` compiles to a bare ``-D X``; ``"None"`` does not."""
    flag = _build_key("f", {"X": None}, "single", "cpu")
    text = _build_key("f", {"X": "None"}, "single", "cpu")
    assert flag != text


def test_an_explicit_disable_survives_a_non_forcing_enable(fake_bempp):
    """configure_opencl enables on every device setup; opting out must stick."""
    module, _calls = fake_bempp
    original = module.build_program

    enable_opencl_program_cache()
    disable_opencl_program_cache()

    assert enable_opencl_program_cache(force=False) is False
    assert module.build_program is original

    assert enable_opencl_program_cache() is True
    assert module.build_program is not original


def test_enable_is_idempotent_and_disable_restores_stock(fake_bempp):
    module, calls = fake_bempp
    original = module.build_program

    assert enable_opencl_program_cache() is True
    patched = module.build_program
    assert enable_opencl_program_cache() is True
    assert module.build_program is patched, "re-enabling must not double-wrap"

    assert disable_opencl_program_cache() is True
    assert module.build_program is original
    assert opencl_program_cache_size() == 0
    assert disable_opencl_program_cache() is False


@pytest.mark.parametrize(
    "replacement",
    [
        # Wrong arity.
        lambda kernel, flags: "program",
        # Extra required parameter the wrapper would never pass.
        lambda assembly_function, options, precision, device_type, extra: "p",
        # Right names, but device_type became keyword-only: a name-only check
        # passes this and then the wrapper's positional call fails.
        eval(
            "lambda assembly_function, options, precision, *, device_type='cpu': 'p'"
        ),
        # Right names in the wrong order.
        lambda options, assembly_function, precision, device_type="cpu": "p",
    ],
    ids=["wrong-arity", "extra-required", "keyword-only", "reordered"],
)
def test_unexpected_upstream_signature_leaves_bempp_untouched(
    fake_bempp, replacement,
):
    """A changed bempp-cl signature must disable the cache, not guess."""
    module, _calls = fake_bempp

    module.build_program = replacement
    assert enable_opencl_program_cache() is False
    assert module.build_program is replacement


def test_a_different_context_is_a_different_cache_entry(fake_bempp):
    """A kernel belongs to the context it was built against."""
    module, calls = fake_bempp
    enable_opencl_program_cache()
    options = {"KERNEL_FUNCTION": "helmholtz_double_layer", "VEC_LENGTH": 8}

    module.build_program("dense_regular_vec", options, "single", "cpu")
    # set_default_cpu_device builds a new context of the *same* device type.
    module._contexts["cpu"] = _FakeContext(0x9999)
    module.build_program("dense_regular_vec", options, "single", "cpu")

    assert len(calls) == 2, "a new context must not be served the old program"


def test_a_changed_include_path_is_a_different_cache_entry(fake_bempp):
    module, calls = fake_bempp
    enable_opencl_program_cache()
    options = {"KERNEL_FUNCTION": "helmholtz_double_layer", "VEC_LENGTH": 8}

    module.build_program("dense_regular_vec", options, "single", "cpu")
    module._INCLUDE_PATH = r'"D:\other\include"'
    module.build_program("dense_regular_vec", options, "single", "cpu")

    assert len(calls) == 2


def test_the_cache_is_bounded(fake_bempp, monkeypatch):
    """A long-lived server meeting many mesh shapes must not grow forever."""
    import hornlab_bempp_bem._opencl_program_cache as cache_module

    monkeypatch.setattr(cache_module, "_MAX_CACHED_PROGRAMS", 4)
    module, _calls = fake_bempp
    enable_opencl_program_cache()

    for index in range(20):
        module.build_program(
            "dense_regular_vec",
            {"KERNEL_FUNCTION": f"kernel_{index}", "VEC_LENGTH": 8},
            "single",
            "cpu",
        )

    assert opencl_program_cache_size() == 4


def test_the_cache_is_per_thread(fake_bempp):
    """pyopencl Kernels are mutable, so threads must not share one."""
    import threading

    module, calls = fake_bempp
    enable_opencl_program_cache()
    options = {"KERNEL_FUNCTION": "helmholtz_double_layer", "VEC_LENGTH": 8}

    module.build_program("dense_regular_vec", options, "single", "cpu")
    other = []
    thread = threading.Thread(
        target=lambda: other.append(
            module.build_program("dense_regular_vec", options, "single", "cpu")
        )
    )
    thread.start()
    thread.join()

    assert len(calls) == 2, "a second thread must build its own kernel"
    assert other and other[0] != "program::dense_regular_vec::1"


def test_clear_reaches_other_threads(fake_bempp):
    import threading

    module, calls = fake_bempp
    enable_opencl_program_cache()
    options = {"KERNEL_FUNCTION": "helmholtz_double_layer", "VEC_LENGTH": 8}

    sizes = []

    def worker(before_clear):
        module.build_program("dense_regular_vec", options, "single", "cpu")
        if before_clear:
            clear_opencl_program_cache()
        sizes.append(opencl_program_cache_size())

    module.build_program("dense_regular_vec", options, "single", "cpu")
    assert opencl_program_cache_size() == 1

    thread = threading.Thread(target=worker, args=(True,))
    thread.start()
    thread.join()

    assert opencl_program_cache_size() == 0, "clear must reach this thread too"


def test_missing_bempp_is_not_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "bempp_cl.core.opencl_kernels", None)
    assert enable_opencl_program_cache() is False
