"""The configured algorithm guards allow data dependencies and reject runtimes."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from grimp import ImportGraph
from importlinter.configuration import configure
from importlinter.contracts.forbidden import ForbiddenContract


def _contract(prefix: str) -> ForbiddenContract:
    path = Path(__file__).resolve().parents[3] / "pyproject.toml"
    options = tomllib.loads(path.read_text())["tool"]["importlinter"]
    definition = next(c for c in options["contracts"] if c["name"].startswith(prefix))
    # import-linter's option reader normalizes TOML booleans to strings.
    definition = {
        key: str(value).lower() if isinstance(value, bool) else value
        for key, value in definition.items()
    }
    configure()
    return ForbiddenContract(
        name=definition["name"], session_options=options, contract_options=definition
    )


def _graph(*edges: tuple[str, str]) -> ImportGraph:
    graph = ImportGraph()
    modules = {
        "everos",
        "everos.algo",
        "everos.algo.rank",
        "everos.component",
        "everos.config",
        "everos.core",
        "everos.entrypoints",
        "everos.infra",
        "everos.memory",
        "everos.service",
    }
    modules.update(module for edge in edges for module in edge)
    for module in modules:
        graph.add_module(module)
    for importer, imported in edges:
        graph.add_import(
            importer=importer,
            imported=imported,
            line_number=1,
            line_contents=f"import {imported}",
        )
    return graph


@pytest.mark.parametrize(
    "target", ["everos.memory", "everos.config", "everos.infra", "everos.core"]
)
def test_algo_cannot_reach_host_layers_even_through_a_helper(target: str) -> None:
    graph = _graph(
        ("everos.algo.rank", "everos.algo.helper"),
        ("everos.algo.helper", target),
    )
    assert not _contract("algo-independent:").check(graph, verbose=False).kept


def test_data_and_internal_algorithm_dependencies_are_allowed() -> None:
    graph = _graph(
        ("everos.algo.rank", "everos.algo.models"),
        ("everos.algo.rank", "pydantic"),
        ("everos.algo.rank", "tiktoken"),
    )
    assert _contract("algo-independent:").check(graph, verbose=False).kept
    assert _contract("algo-no-runtime:").check(graph, verbose=False).kept


@pytest.mark.parametrize("target", ["os", "sqlite3", "openai", "lancedb"])
def test_algo_cannot_import_runtime_implementations(target: str) -> None:
    graph = _graph(("everos.algo.rank", target))
    assert not _contract("algo-no-runtime:").check(graph, verbose=False).kept
