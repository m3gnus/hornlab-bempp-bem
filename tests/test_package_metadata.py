from __future__ import annotations

from pathlib import Path
import re


def test_default_opencl_backend_declares_pyopencl_dependency():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject_text = pyproject.read_text(encoding="utf-8")
    dependency_block = re.search(r"dependencies\s*=\s*\[(?P<deps>.*?)\]", pyproject_text, re.S)
    assert dependency_block is not None
    deps = re.findall(r'"([^"]+)"', dependency_block.group("deps"))
    names = {
        re.split(r"[<>=!~;\[]", dependency, maxsplit=1)[0].strip().lower().replace("_", "-")
        for dependency in deps
    }

    assert "pyopencl" in names
