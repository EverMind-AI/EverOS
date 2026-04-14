"""
Task 3: sandbox path builder + sync/async Node bridge wrappers.

The bridge stub used here is the real echo stub at
evaluation/scripts/openclaw_eval_bridge.mjs; these tests double as an
integration check that node + subprocess plumbing works end-to-end.
"""
from pathlib import Path

import pytest

from evaluation.src.adapters.openclaw_runtime import (
    BridgeError,
    BridgeTimeout,
    arun_bridge,
    build_sandbox_paths,
    run_bridge,
)


BRIDGE_PATH = (
    Path(__file__).parents[2]
    / "evaluation"
    / "scripts"
    / "openclaw_eval_bridge.mjs"
)


def test_build_sandbox_paths_isolated_per_conversation(tmp_path):
    paths = build_sandbox_paths(tmp_path, run_id="run-1", conversation_id="locomo_0")
    assert Path(paths["workspace_dir"]).name == "locomo_0"
    assert Path(paths["native_store_dir"]).parent.name == "locomo_0"
    assert Path(paths["metrics_dir"]).name == "metrics"
    assert Path(paths["events_path"]).name == "events.jsonl"
    # base_dir points at the conversation sandbox root
    assert Path(paths["base_dir"]).name == "locomo_0"


def test_build_sandbox_paths_run_id_segregation(tmp_path):
    a = build_sandbox_paths(tmp_path, run_id="run-A", conversation_id="c0")
    b = build_sandbox_paths(tmp_path, run_id="run-B", conversation_id="c0")
    assert Path(a["base_dir"]) != Path(b["base_dir"])


def test_run_bridge_returns_stub_response():
    payload = {"command": "status", "workspace_dir": "/tmp/x"}
    result = run_bridge(BRIDGE_PATH, payload)
    assert result["ok"] is True
    assert result["command"] == "status"


@pytest.mark.asyncio
async def test_arun_bridge_returns_stub_response():
    payload = {"command": "status", "workspace_dir": "/tmp/x"}
    result = await arun_bridge(BRIDGE_PATH, payload)
    assert result["ok"] is True
    assert result["command"] == "status"


def test_run_bridge_raises_on_missing_script(tmp_path):
    missing = tmp_path / "does_not_exist.mjs"
    with pytest.raises(BridgeError):
        run_bridge(missing, {"command": "status"})


def test_run_bridge_raises_on_ok_false(tmp_path):
    failing = tmp_path / "fail.mjs"
    failing.write_text(
        'process.stdout.write(JSON.stringify({"ok": false, "error": "boom"}));\n'
    )
    with pytest.raises(BridgeError, match="boom"):
        run_bridge(failing, {"command": "status"})


def test_run_bridge_raises_on_invalid_json(tmp_path):
    bad = tmp_path / "bad.mjs"
    bad.write_text('process.stdout.write("not json at all");\n')
    with pytest.raises(BridgeError, match="invalid json"):
        run_bridge(bad, {"command": "status"})


def test_run_bridge_raises_on_timeout(tmp_path):
    hang = tmp_path / "hang.mjs"
    hang.write_text(
        'await new Promise((r) => setTimeout(r, 60000));\n'
        'process.stdout.write(JSON.stringify({"ok": true}));\n'
    )
    with pytest.raises(BridgeTimeout):
        run_bridge(hang, {"command": "status"}, timeout=0.5)
