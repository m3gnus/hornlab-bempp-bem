"""A failed device switch must preserve the last usable Bempp selection."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from hornlab_bempp_bem import _opencl_program_cache, _opencl_singular_workgroup, device


@pytest.fixture
def runtime(monkeypatch):
    names = (
        "pyopencl",
        "bempp_cl",
        "bempp_cl.api",
        "bempp_cl.core",
        "bempp_cl.core.opencl_kernels",
    )
    modules = {name: ModuleType(name) for name in names}
    api, core, kernels = (modules[name] for name in names[2:])
    modules["bempp_cl"].api = api
    modules["bempp_cl"].core = core
    core.opencl_kernels = kernels
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for kind in ("cpu", "gpu"):
        setattr(
            kernels,
            f"default_{kind}_device",
            Mock(
                return_value=SimpleNamespace(name=kind.upper()),
            ),
        )
        setattr(kernels, f"default_{kind}_context", Mock(return_value=object()))
    for name in ("enable_opencl_build_workarounds", "enable_opencl_program_cache"):
        monkeypatch.setattr(_opencl_program_cache, name, Mock(return_value=True))
    clear = Mock()
    monkeypatch.setattr(_opencl_program_cache, "clear_opencl_program_cache", clear)
    monkeypatch.setattr(
        _opencl_singular_workgroup,
        "enable_singular_workgroup_tuning",
        Mock(),
    )
    monkeypatch.setattr(device, "_active_device_type", None)
    monkeypatch.setattr(device, "_active_device_name", None)
    return api, kernels, clear


@pytest.mark.parametrize(
    "initial,target,failed_call",
    [
        ("cpu", "gpu", "default_gpu_device"),
        ("cpu", "gpu", "default_gpu_context"),
        ("cpu", "gpu", "default_cpu_device"),
        ("cpu", "gpu", "default_cpu_context"),
        ("gpu", "cpu", "default_cpu_device"),
        ("gpu", "cpu", "default_cpu_context"),
    ],
)
def test_failed_switch_preserves_previous_selection(
    runtime, initial, target, failed_call
):
    api, kernels, clear = runtime
    assert device.configure_opencl(initial) == initial.upper()
    failure = getattr(kernels, failed_call)
    failure.side_effect = RuntimeError("device unavailable")

    with pytest.raises(device.OpenCLError, match="could not be initialized"):
        device.configure_opencl(target)

    assert device.configure_opencl(initial) == initial.upper()
    assert api.BOUNDARY_OPERATOR_DEVICE_TYPE == initial
    assert api.POTENTIAL_OPERATOR_DEVICE_TYPE == initial
    clear.assert_not_called()

    failure.side_effect = None
    assert device.configure_opencl(target) == target.upper()
    assert api.BOUNDARY_OPERATOR_DEVICE_TYPE == target
    assert api.POTENTIAL_OPERATOR_DEVICE_TYPE == target
    clear.assert_called_once_with()
