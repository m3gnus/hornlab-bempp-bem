from __future__ import annotations

from pathlib import Path


def test_default_opencl_backend_declares_pyopencl_dependency():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"

    assert '"pyopencl"' in pyproject.read_text(encoding="utf-8")
