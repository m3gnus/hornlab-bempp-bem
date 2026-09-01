"""The OpenCL preflight check must fail when the runtime is *broken*, not just absent.

The cheap alternatives -- enumerating ``pyopencl.get_platforms()`` for a
CPU-type device, or reading back ``bempp_cl.api.DEFAULT_DEVICE_INTERFACE`` --
both pass on a machine whose device enumerates fine but whose kernels will not
compile. That case is real here: ``_opencl_program_cache`` carries a workaround
for exactly it, an install path containing a space. So the check assembles a
real operator, and these tests pin that it reports the stage that failed.
"""
from __future__ import annotations

import pytest

import hornlab_bempp_bem as bempp_bem
from hornlab_bempp_bem.device import (
    OpenCLCheck,
    OpenCLError,
    check_opencl,
    require_opencl,
)


def test_check_opencl_reports_ok_on_a_machine_with_a_working_runtime():
    result = check_opencl("cpu")
    if not result.ok and result.stage in {"pyopencl", "device"}:
        pytest.skip(f"no usable OpenCL device here: {result.detail}")

    assert result.ok is True
    assert result.stage == "ok"
    assert result.device_name
    assert result.device_name == result.device_name.strip()
    assert result.detail is None
    assert "OpenCL OK" in result.describe()


def test_a_missing_device_is_reported_at_the_device_stage(monkeypatch):
    """No device at all: stop before assembly and say which stage failed."""

    def no_device(device_type="cpu"):
        raise OpenCLError("Could not find suitable OpenCL CPU driver")

    monkeypatch.setattr("hornlab_bempp_bem.device.configure_opencl", no_device)

    result = check_opencl("cpu")
    assert result.ok is False
    assert result.stage == "device"
    assert result.device_name is None
    assert "CPU driver" in (result.detail or "")
    assert "UNAVAILABLE" in result.describe()


def test_a_present_but_broken_runtime_is_reported_at_the_assembly_stage(monkeypatch):
    """The case the cheap checks miss: device enumerates, kernel will not build.

    Enumeration and ``DEFAULT_DEVICE_INTERFACE`` both look healthy here, so
    only a check that actually assembles can catch it.
    """
    monkeypatch.setattr(
        "hornlab_bempp_bem.device.configure_opencl",
        lambda device_type="cpu": "Pretend CPU Device",
    )

    from bempp_cl.api.operators.boundary import helmholtz

    def broken_build(*args, **kwargs):
        raise RuntimeError("clBuildProgram failed: CL_BUILD_PROGRAM_FAILURE")

    monkeypatch.setattr(helmholtz, "single_layer", broken_build)

    result = check_opencl("cpu")
    assert result.ok is False
    assert result.stage == "assembly"
    # The device name survives, because the device really was there.
    assert result.device_name == "Pretend CPU Device"
    assert "clBuildProgram" in (result.detail or "")


def test_require_opencl_raises_with_the_stage_in_the_message(monkeypatch):
    monkeypatch.setattr(
        "hornlab_bempp_bem.device.configure_opencl",
        lambda device_type="cpu": (_ for _ in ()).throw(OpenCLError("no driver")),
    )

    with pytest.raises(OpenCLError) as excinfo:
        require_opencl("cpu")
    assert "device" in str(excinfo.value)


def test_require_opencl_returns_the_check_when_healthy():
    try:
        result = require_opencl("cpu")
    except OpenCLError as exc:
        pytest.skip(f"no usable OpenCL device here: {exc}")
    assert isinstance(result, OpenCLCheck)
    assert result.ok is True


def test_the_check_does_not_use_the_deprecated_operator_A_property():
    """bempp-cl 0.4.2 deprecates ``operator.A``; ``to_dense()`` is the same call.

    ``.A`` is a property that warns and then returns ``to_dense()``, so this is
    a pure deprecation fix with no numerical consequence. Pinned because the
    check is the one place in this package that densifies an operator directly,
    and a DeprecationWarning from a startup preflight is noise a user cannot act
    on.
    """
    import inspect

    from hornlab_bempp_bem import device

    source = inspect.getsource(device.check_opencl)
    assert ".to_dense()" in source
    assert ".weak_form().A" not in source


def test_the_check_is_exported_from_the_package():
    """It is a startup-time entry point, so it belongs in the public surface."""
    for name in ("OpenCLCheck", "check_opencl", "require_opencl"):
        assert name in bempp_bem.__all__
        assert hasattr(bempp_bem, name)
