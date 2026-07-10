from __future__ import annotations

from pathlib import Path
import re


def test_pyopencl_is_an_opencl_extra_not_a_base_dependency():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject_text = pyproject.read_text(encoding="utf-8")
    dependency_block = re.search(r"dependencies\s*=\s*\[(?P<deps>.*?)\]", pyproject_text, re.S)
    assert dependency_block is not None
    deps = re.findall(r'"([^"]+)"', dependency_block.group("deps"))
    names = {
        re.split(r"[<>=!~;\[]", dependency, maxsplit=1)[0].strip().lower().replace("_", "-")
        for dependency in deps
    }

    assert "pyopencl" not in names

    opencl_extra = re.search(
        r"^opencl\s*=\s*\[(?P<deps>.*?)\]",
        pyproject_text,
        re.S | re.M,
    )
    assert opencl_extra is not None
    extra_names = {
        re.split(r"[<>=!~;\[]", dependency, maxsplit=1)[0]
        .strip()
        .lower()
        .replace("_", "-")
        for dependency in re.findall(r'"([^"]+)"', opencl_extra.group("deps"))
    }
    assert "pyopencl" in extra_names
