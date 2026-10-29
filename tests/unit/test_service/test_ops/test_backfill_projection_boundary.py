"""Keep the backfill facade exception limited to pure projection declarations."""

from __future__ import annotations

import ast
import importlib.util
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_MODULE = "everos.service.ops._backfill_vectors"
_FACADE = "everos.infra.persistence.index"
_RUNTIME_NAMES = {"VectorProjection", "VECTOR_PROJECTIONS"}


def _infra_import_violations(source: str) -> list[str]:
    """Check static imports, including aliases, relative imports and annotations."""
    violations: list[str] = []

    def visit(node: ast.AST, *, annotation_only: bool = False) -> None:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for child in node.body:
                visit(child, annotation_only=True)
            for child in node.orelse:
                visit(child, annotation_only=annotation_only)
            return
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = importlib.util.resolve_name(
                    "." * node.level + module, _MODULE.rpartition(".")[0]
                )
            allowed = _RUNTIME_NAMES | (
                {"IndexMaintenance"} if annotation_only else set()
            )
            for alias in node.names:
                target = f"{module}.{alias.name}"
                if target.startswith("everos.infra.") and (
                    module != _FACADE or alias.name not in allowed
                ):
                    violations.append(target)
        elif isinstance(node, ast.Import):
            violations.extend(
                alias.name
                for alias in node.names
                if alias.name == "everos.infra"
                or alias.name.startswith("everos.infra.")
            )
        for child in ast.iter_child_nodes(node):
            visit(child, annotation_only=annotation_only)

    visit(ast.parse(source))
    return violations


def test_backfill_imports_only_projection_declarations() -> None:
    source = _ROOT / "src/everos/service/ops/_backfill_vectors.py"
    assert _infra_import_violations(source.read_text()) == []


def test_import_linter_exception_is_exact() -> None:
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    contract = next(
        item
        for item in config["tool"]["importlinter"]["contracts"]
        if item["name"].startswith("service-usecases-forbidden:")
    )
    exceptions = [item for item in contract["ignore_imports"] if _MODULE in item]
    assert exceptions == [f"{_MODULE} -> {_FACADE}"]
    assert "everos.infra" in contract["forbidden_modules"]
    backend_guard = next(
        item
        for item in config["tool"]["importlinter"]["contracts"]
        if item["name"] == "Derived-index ports know no concrete backend"
    )
    assert f"{_FACADE}.vector_projection" in backend_guard["source_modules"]


@pytest.mark.parametrize(
    "statement",
    [
        "from everos.infra.persistence.index import DerivedIndex",
        "from everos.infra.persistence.index import IndexBackend as Factory",
        "from everos.infra.persistence.index import IndexMaintenance",
        "from everos.infra.persistence.index import *",
        "import everos.infra.persistence.index as index",
        "from everos.infra.persistence import index",
        "from everos.infra.persistence.factories import build_derived_index",
        "from ...infra.persistence.index import DerivedIndex",
        "from everos.infra.persistence.index.vector_projection import VectorProjection",
        "if TYPE_CHECKING:\n"
        "    from everos.infra.persistence.index import DerivedIndex",
    ],
)
def test_exception_rejects_backend_and_module_access(statement: str) -> None:
    assert _infra_import_violations(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "from everos.infra.persistence.index import "
        "VectorProjection, VECTOR_PROJECTIONS",
        "from everos.infra.persistence.index import VectorProjection as Projection",
        "if TYPE_CHECKING:\n"
        "    from everos.infra.persistence.index import IndexMaintenance",
    ],
)
def test_exception_accepts_pure_declarations(statement: str) -> None:
    assert _infra_import_violations(statement) == []
