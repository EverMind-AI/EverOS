"""Keep the vectors phase free of runtime infra imports, with no exception.

The row-level backfill lives in
:mod:`everos.infra.persistence.index.vector_backfill`; the phase receives it
through :class:`~everos.service.BackfillDeps`. A type import under
``TYPE_CHECKING`` is not a dependency (architecture rule 4); anything else
reaching ``everos.infra`` is, and must not come back as an ``ignore_imports``
line in the ``service-usecases-forbidden`` contract.
"""

from __future__ import annotations

import ast
import importlib.util
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_MODULE = "everos.service.ops._backfill_vectors"


def _runtime_infra_imports(source: str) -> list[str]:
    """Static ``everos.infra`` imports outside ``if TYPE_CHECKING:`` blocks.

    Covers aliases, relative imports and star imports.
    """
    violations: list[str] = []

    def visit(node: ast.AST) -> None:
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Name)
            and node.test.id == "TYPE_CHECKING"
        ):
            for child in node.orelse:
                visit(child)
            return
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                module = importlib.util.resolve_name(
                    "." * node.level + module, _MODULE.rpartition(".")[0]
                )
            violations.extend(
                f"{module}.{alias.name}"
                for alias in node.names
                if module == "everos.infra" or module.startswith("everos.infra.")
            )
        elif isinstance(node, ast.Import):
            violations.extend(
                alias.name
                for alias in node.names
                if alias.name == "everos.infra"
                or alias.name.startswith("everos.infra.")
            )
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(ast.parse(source))
    return violations


def test_vectors_phase_has_no_runtime_infra_import() -> None:
    source = _ROOT / "src/everos/service/ops/_backfill_vectors.py"
    assert _runtime_infra_imports(source.read_text()) == []


def test_import_linter_carries_no_backfill_exception() -> None:
    config = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    contract = next(
        item
        for item in config["tool"]["importlinter"]["contracts"]
        if item["name"].startswith("service-usecases-forbidden:")
    )
    assert "everos.infra" in contract["forbidden_modules"]
    assert [
        item for item in contract["ignore_imports"] if "_backfill_vectors" in item
    ] == []


@pytest.mark.parametrize(
    "statement",
    [
        "from everos.infra.persistence.index import VectorProjection",
        "from everos.infra.persistence.index import VECTOR_PROJECTIONS as P",
        "from everos.infra.persistence.index import *",
        "from everos.infra import persistence",
        "import everos.infra.persistence.index as index",
        "from ...infra.persistence.index import DerivedIndex",
        "if TYPE_CHECKING:\n    pass\nelse:\n"
        "    from everos.infra.persistence.index import DerivedIndex",
    ],
)
def test_checker_rejects_runtime_infra_imports(statement: str) -> None:
    assert _runtime_infra_imports(statement)


@pytest.mark.parametrize(
    "statement",
    [
        "if TYPE_CHECKING:\n"
        "    from everos.infra.persistence.index import VectorBackfill",
        "from everos.service.ops._backfill import BackfillDeps",
        "from everos.memory.ports import Filter",
    ],
)
def test_checker_accepts_type_only_and_non_infra_imports(statement: str) -> None:
    assert _runtime_infra_imports(statement) == []
