from __future__ import annotations

import tomllib
from pathlib import Path


def _all_specs_are_exact(specs: list[str]) -> bool:
    return all("==" in spec and not any(op in spec for op in (">=", "<=", "~=", ">", "<")) for spec in specs)


def test_project_dependencies_are_exactly_pinned():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    assert isinstance(dependencies, list)
    assert _all_specs_are_exact(dependencies)


def test_build_system_dependencies_are_exactly_pinned():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    build_requires = pyproject["build-system"]["requires"]
    assert isinstance(build_requires, list)
    assert _all_specs_are_exact(build_requires)
