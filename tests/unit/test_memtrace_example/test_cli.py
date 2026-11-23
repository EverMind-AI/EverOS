"""CLI listing is offline and reports never overwrite input data or credentials."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
from smartcomment.runtime import ExecNetwork

from examples.memtrace import attribute as cli
from examples.memtrace.merge import merge_graphs


def _argv(monkeypatch: pytest.MonkeyPatch, path: Path, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["attribute.py", "--graph", str(path), *args])


def test_listing_needs_no_credentials_or_client(
    graph: ExecNetwork,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph.export_graph()), encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    _argv(monkeypatch, path, "--list-cases")
    assert cli.main() == 0
    assert json.loads(capsys.readouterr().out)[0]["query"] == (
        "When was the adoption meeting?"
    )


@pytest.mark.parametrize("target", ["graph", "credentials"])
def test_report_cannot_overwrite_input_or_credentials(
    graph: ExecNetwork,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph.export_graph()), encoding="utf-8")
    credentials = tmp_path / "credentials.json"
    credentials.write_text(
        '{"api_keys": ["fake-key"], "base_urls": ["https://example.invalid"]}',
        encoding="utf-8",
    )
    output = path if target == "graph" else credentials
    before = output.read_bytes()
    _argv(monkeypatch, path, "--output", str(output), "--api-config", str(credentials))
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert output.read_bytes() == before


@pytest.mark.parametrize("successful", [True, False])
def test_cli_writes_report_and_returns_status(
    graph: ExecNetwork,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    successful: bool,
) -> None:
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph.export_graph()), encoding="utf-8")
    original = path.read_bytes()
    output = tmp_path / "reports" / "report.json"
    monkeypatch.setenv("OPENAI_API_KEY", "fake-unit-secret")
    monkeypatch.setenv("OPENAI_MODEL", "unit-model")

    async def fake_run(*args: Any) -> dict[str, Any]:
        return {
            "status": "success" if successful else "failed",
            "attribution": None,
            "failure": None if successful else {"type": "InsufficientEvidence"},
            "iterations": 1,
            "usage": {},
        }

    monkeypatch.setattr(cli, "_run", fake_run)
    _argv(monkeypatch, path, "--query-id", "query", "--output", str(output))
    assert cli.main() == (0 if successful else 1)
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["diagnostics"] == []
    assert "fake-unit-secret" not in output.read_text(encoding="utf-8")
    assert path.read_bytes() == original


def test_missing_graph_is_a_setup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _argv(monkeypatch, tmp_path / "missing.json", "--list-cases")
    assert cli.main() == 2


def test_conflicting_legacy_merge_is_rejected_before_credentials(
    graph: ExecNetwork,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = graph.export_graph()
    payload["metadata"] = {"merge_value_conflicts": ["memory@1"]}
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _argv(monkeypatch, path, "--list-cases")
    assert cli.main() == 2
    assert "unresolved revision conflicts" in capsys.readouterr().err


def test_safe_merged_query_can_be_selected_by_original_identity(
    graph: ExecNetwork,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = graph.export_graph()
    payload["metadata"] = {"app_id": "a", "project_id": "p"}
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(merge_graphs([payload])), encoding="utf-8")
    output = tmp_path / "report.json"
    monkeypatch.setenv("OPENAI_API_KEY", "fake-unit-key")

    async def fake_run(graph: ExecNetwork, case: Any, *args: Any) -> dict[str, Any]:
        assert "/query@1" in case.query_full_node_id
        return {
            "status": "failed",
            "iterations": 0,
            "usage": {},
            "attribution": None,
            "failure": {"type": "InsufficientEvidence"},
        }

    monkeypatch.setattr(cli, "_run", fake_run)
    _argv(monkeypatch, path, "--query-id", "query", "--output", str(output))
    assert cli.main() == 1
