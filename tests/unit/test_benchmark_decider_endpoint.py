"""A decider that cannot answer must stop the run, not quietly change the experiment.

Regression cover for a defect that voided a whole batch of runs at once. That batch
served each candidate decider on its own endpoint and passed
``--decider-model policy``; ``--decider-base-url`` did not exist, so the new model name
went to the endpoint the config named -- a gateway with no model called ``policy``.

Nothing failed. ``llm_multiround`` retries the decider four times, then falls back
to a deterministic top-3 core and stops after one round (``_DECIDER_FALLBACK_CORE``),
so every arm produced a complete report: 87-93%, plausible, and near-identical across
four model sizes because all four ran the same fallback. The traces are unambiguous --
0 successful decider calls out of 12,580 rounds, ``core_indices == [0, 1, 2]`` every
time, and zero requests ever reached the checkpoint servers.

Pinned here: the endpoint is settable alongside the model; the servers get the decider
as ``EVEROS_LLM__DECIDER__*`` from the same environment builder the probe uses; the
probe resolves and asks the decider through EverOS's own ``resolve_llm_task`` and
provider, so its request is the decider's request; a decider that does not answer
raises before the first question; the key appears in no output; and the run record
names the decider the servers resolve.
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

run = importlib.import_module("run")
decider_probe = importlib.import_module("decider_probe")
BenchmarkConfig = importlib.import_module("config").BenchmarkConfig

_NO_THINK = '{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'
_KEY = "sk-decider-SECRET-0123456789"
_UNREACHABLE = "http://127.0.0.1:9/v1"


class _FakeDecider:
    """An OpenAI-compatible endpoint that records each request and replies as told.

    Attributes:
        requests: Every request body received, parsed.
        auth: The ``Authorization`` header of each request.
        status: HTTP status of the reply.
        content: The reply's message content; ``None`` sends no content.
        finish_reason: The reply's finish reason.
        error_message: The error body's message when ``status`` is not 200.
    """

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.auth: list[str] = []
        self.status = 200
        self.content: str | None = '{"ready": true}'
        self.finish_reason = "stop"
        self.error_message = ""
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        threading.Thread(
            target=self._server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        ).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _reply(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.status != 200:
            return {"error": {"message": self.error_message}}
        return {
            "id": "probe",
            "object": "chat.completion",
            "created": 0,
            "model": request.get("model", ""),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.content},
                    "finish_reason": self.finish_reason,
                }
            ],
        }

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                request = json.loads(self.rfile.read(length))
                fake.requests.append(request)
                fake.auth.append(self.headers.get("Authorization", ""))
                body = json.dumps(fake._reply(request)).encode()
                self.send_response(fake.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return None

        return Handler


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No LLM setting or proxy from the developer's shell reaches a probe."""
    for name in list(os.environ):
        if name.startswith(("EVEROS_LLM__", "EVEROS_DECIDER__", "EVEROS_LLMMR_")) or (
            name.lower().endswith("_proxy")
        ):
            monkeypatch.delenv(name)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")


@pytest.fixture
def fake() -> Iterator[_FakeDecider]:
    server = _FakeDecider()
    yield server
    server.close()


def _cfg(**over: Any) -> Any:
    """A multi-round config; the decider fields live in the server environment."""
    return BenchmarkConfig(methods="llm_multiround").model_copy(update=over)


def _server_env(root: Path, models: Any, extra: dict[str, str] | None = None) -> dict:
    return run.server_launch_env(os.environ, root, models, extra or {})


def _probe(**over: Any) -> Any:
    fields: dict[str, Any] = {
        "model": "policy",
        "base_url": "http://127.0.0.1:9360/v1",
        "host": "127.0.0.1",
        "sources": {"model": "llm.decider", "base_url": "llm.decider"},
        "outcome": "answered",
        "reason": "",
        "error": "",
    }
    fields.update(over)
    return run.DeciderProbe(**fields)


def _set_decider(monkeypatch: pytest.MonkeyPatch, base_url: str, **fields: str) -> None:
    """``[llm.decider]`` through its variables, as the fleet exports it."""
    monkeypatch.setenv("EVEROS_LLM__DECIDER__MODEL", "policy")
    monkeypatch.setenv("EVEROS_LLM__DECIDER__BASE_URL", base_url)
    monkeypatch.setenv("EVEROS_LLM__DECIDER__API_KEY", _KEY)
    for name, value in fields.items():
        monkeypatch.setenv(f"EVEROS_LLM__DECIDER__{name.upper()}", value)


# ---------------------------------------------------------------------------
# The flags and the servers' environment
# ---------------------------------------------------------------------------


def test_decider_base_url_is_a_flag() -> None:
    """The model can be overridden; so must the endpoint that serves it.

    With only ``--decider-model``, changing the decider points a new name at the old
    endpoint -- which is the whole defect, in one argument.
    """
    src = (_BENCH / "run.py").read_text(encoding="utf-8")
    assert '"--decider-base-url"' in src
    assert '"--decider-api-key"' in src


def test_overrides_reach_the_config_not_just_the_fleet() -> None:
    """One source of truth, or provenance lies.

    The old code passed ``args.decider_model or config.decider_model`` to the server
    fleet while ``run_spec.json`` kept reporting ``config.decider_model``. Every arm was
    therefore filed as ``decider = qwen3.6-27B`` regardless of what it ran.
    """
    src = (_BENCH / "run.py").read_text(encoding="utf-8")
    assert '_over["decider_base_url"] = args.decider_base_url' in src
    assert "decider_model=config.decider_model," in src
    assert "args.decider_model or config.decider_model" not in src


def test_the_fleet_exports_the_decider_as_llm_decider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The servers get ``[llm.decider]``, and exactly what the probe resolves under.

    ``EVEROS_DECIDER__*`` still works server-side but logs a deprecation warning, so the
    fleet no longer exports it. The probe's environment comes from the same builder
    with the same inputs; the only thing it leaves out is the per-server trace file.
    """
    envs: list[dict[str, str]] = []

    class _Proc:
        def __init__(self, *_a: Any, **kw: Any) -> None:
            envs.append(kw["env"])

        def poll(self) -> None:
            return None

    class _NoWaitFleet(run._ServerFleet):
        def _await_ready(self, timeout_s: float = 180.0) -> None:
            return None

    monkeypatch.setattr(run.subprocess, "Popen", _Proc)
    root = tmp_path / "store"
    root.mkdir()
    (root / "everos.toml").write_text("", encoding="utf-8")
    models = run.ServerModels(
        backbone_model="backbone",
        decider_model="policy",
        decider_base_url="http://127.0.0.1:9360/v1",
        decider_api_key=_KEY,
    )
    extra = {"EVEROS_CASCADE__ENABLED": "false"}
    fleet = _NoWaitFleet(1, root, first_port=0, models=models, extra_env=extra)
    fleet.start()

    (env,) = envs
    assert env["EVEROS_LLM__DECIDER__MODEL"] == "policy"
    assert env["EVEROS_LLM__DECIDER__BASE_URL"] == "http://127.0.0.1:9360/v1"
    assert env["EVEROS_LLM__DECIDER__API_KEY"] == _KEY
    assert env["EVEROS_LLM__MODEL"] == "backbone"
    assert not [name for name in env if name.startswith("EVEROS_DECIDER__")]
    assert env == run.server_launch_env(os.environ, fleet.roots[0], models, extra)
    assert _KEY not in repr(models), "the key must not show up in a repr"


def test_the_fleet_and_the_probe_share_one_environment_builder() -> None:
    """Two hand-kept copies of the launch environment drift; one function cannot."""
    assert "server_launch_env(" in inspect.getsource(run._ServerFleet._start_once)
    assert "server_launch_env(" in inspect.getsource(run.main)


def test_a_trace_file_is_added_only_when_given(tmp_path: Path) -> None:
    models = run.ServerModels()
    env = run.server_launch_env({}, tmp_path, models, {}, trace_file=tmp_path / "t")
    assert env == {
        "EVEROS_ROOT": str(tmp_path),
        "EVEROS_LLMMR_TRACE_DUMP": str(tmp_path / "t"),
    }
    assert run.server_launch_env({}, tmp_path, models, {}) == {
        "EVEROS_ROOT": str(tmp_path)
    }


# ---------------------------------------------------------------------------
# The probe's request is the decider's request (in process, against a fake endpoint)
# ---------------------------------------------------------------------------


def test_the_probe_sends_what_llm_decider_configures(
    fake: _FakeDecider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Model, endpoint, key, ``extra`` and output cap all come from the client config.

    A re-run once failed because the probe omitted ``enable_thinking=false`` while
    the servers had it; another vouched for an uncapped call the run never made. The
    probe now has no request fields of its own, so neither can recur. No temperature
    is configured, so none is sent -- the decider's own call sends none either.
    """
    _set_decider(monkeypatch, fake.base_url, extra=_NO_THINK, max_tokens="64")

    report = decider_probe.probe(send=True)

    assert report["outcome"] == "answered", report
    (request,) = fake.requests
    assert request["model"] == "policy"
    assert request["max_tokens"] == 64
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert "temperature" not in request
    assert request["response_format"]["type"] == "json_schema"
    assert request["response_format"]["json_schema"]["strict"] is True
    assert fake.auth == [f"Bearer {_KEY}"]
    assert report["sources"]["extra"] == "llm.decider"
    assert report["sources"]["max_tokens"] == "llm.decider"


def test_the_deprecated_decider_variables_still_reach_the_probe(
    fake: _FakeDecider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch scripts still export ``EVEROS_DECIDER__*``; the factory reads them."""
    monkeypatch.setenv("EVEROS_DECIDER__MODEL", "policy")
    monkeypatch.setenv("EVEROS_DECIDER__BASE_URL", fake.base_url)
    monkeypatch.setenv("EVEROS_DECIDER__API_KEY", _KEY)
    monkeypatch.setenv("EVEROS_DECIDER__EXTRA", _NO_THINK)

    report = decider_probe.probe(send=True)

    assert report["outcome"] == "answered", report
    assert fake.requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert report["sources"]["extra"] == "decider"


def test_what_llm_decider_leaves_unset_comes_from_its_defaults_and_llm(
    fake: _FakeDecider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming only the decider's model inherits ``[llm]``'s endpoint, and is probed.

    The probe once required an explicit decider model *and* URL, so this valid shape
    was never checked -- and an inherited endpoint that does not serve the named model
    fails on the first question rather than at startup.
    """
    monkeypatch.setenv("EVEROS_LLM__MODEL", "backbone")
    monkeypatch.setenv("EVEROS_LLM__BASE_URL", fake.base_url)
    monkeypatch.setenv("EVEROS_LLM__API_KEY", _KEY)
    monkeypatch.setenv("EVEROS_LLM__DECIDER__MODEL", "policy")

    report = decider_probe.probe(send=True)

    assert report["outcome"] == "answered", report
    assert fake.requests[0]["model"] == "policy", "the decider's model, not [llm]'s"
    assert fake.requests[0]["max_tokens"] == 512, "the decider's own default"
    assert report["sources"]["base_url"] == "llm"
    assert report["sources"]["max_tokens"] == "default"


def test_a_configured_temperature_is_sent(
    fake: _FakeDecider, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_decider(monkeypatch, fake.base_url, temperature="0.2")
    decider_probe.probe(send=True)
    assert fake.requests[0]["temperature"] == 0.2


@pytest.mark.parametrize(
    ("content", "finish_reason", "reason"),
    [
        ("", "stop", "invalid_reply"),
        (None, "stop", "no_content"),
        ("ready", "stop", "invalid_reply"),
        ('{"ready": tr', "length", "cut_short"),
    ],
)
def test_an_unusable_reply_is_a_failure(
    fake: _FakeDecider,
    monkeypatch: pytest.MonkeyPatch,
    content: str | None,
    finish_reason: str,
    reason: str,
) -> None:
    """Empty, absent, malformed or cut short: the decider's parser fails all four.

    A reasoning decider that spends its budget thinking returns exactly these, and the
    decider then falls back to the fixed core -- so the probe has to fail them too.
    """
    fake.content, fake.finish_reason = content, finish_reason
    _set_decider(monkeypatch, fake.base_url)
    report = decider_probe.probe(send=True)
    assert (report["outcome"], report["reason"]) == ("failed", reason), report


def test_the_key_is_masked_out_of_the_error(
    fake: _FakeDecider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gateway that echoes the key in its 401 must not put it in the report."""
    fake.status, fake.error_message = 401, f"Incorrect API key provided: {_KEY}"
    _set_decider(monkeypatch, fake.base_url)
    report = decider_probe.probe(send=True)
    assert report["outcome"] == "failed"
    assert "***" in report["error"]
    assert _KEY not in json.dumps(report)


def test_no_usable_endpoint_or_key_is_reported_not_raised() -> None:
    report = decider_probe.probe(send=True)
    assert report["outcome"] == "not_configured", report
    assert report["error"]


# ---------------------------------------------------------------------------
# The child process, end to end
# ---------------------------------------------------------------------------


def test_a_live_decider_passes(
    fake: _FakeDecider,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    models = run.ServerModels(
        decider_model="policy", decider_base_url=fake.base_url, decider_api_key=_KEY
    )
    probe = run._resolve_decider(_cfg(), _server_env(tmp_path, models), ask=True)

    assert probe is not None and probe.outcome == "answered"
    assert (probe.model, probe.base_url, probe.host) == (
        "policy",
        fake.base_url,
        "127.0.0.1",
    )
    assert len(fake.requests) == 1
    out = capsys.readouterr().out
    assert "policy @ 127.0.0.1" in out
    assert _KEY not in out


def test_an_unreachable_decider_stops_the_run(tmp_path: Path) -> None:
    """The production failure: the run must stop, naming the model and the host."""
    models = run.ServerModels(
        decider_model="policy", decider_base_url=_UNREACHABLE, decider_api_key=_KEY
    )
    with pytest.raises(run.DeciderUnreachableError) as caught:
        run._resolve_decider(_cfg(), _server_env(tmp_path, models), ask=True)
    message = str(caught.value)
    assert "policy @ 127.0.0.1" in message
    assert "did not answer" in message
    assert _KEY not in message


def test_an_empty_reply_stops_the_run(fake: _FakeDecider, tmp_path: Path) -> None:
    """A decider that answers with nothing stops the run, and the fix is named."""
    fake.content = ""
    models = run.ServerModels(
        decider_model="policy", decider_base_url=fake.base_url, decider_api_key=_KEY
    )
    with pytest.raises(run.DeciderUnreachableError, match="enable_thinking"):
        run._resolve_decider(_cfg(), _server_env(tmp_path, models), ask=True)


@pytest.mark.parametrize(
    ("status", "content"),
    [
        (401, None),  # the gateway echoes the key in its error
        (200, ""),  # an invalid reply: the provider logs a warning per attempt
    ],
)
def test_the_child_prints_one_json_line_without_the_key(
    fake: _FakeDecider, tmp_path: Path, status: int, content: str | None
) -> None:
    """Stdout carries the report alone -- log lines go to stderr -- and no key."""
    fake.status, fake.content = status, content
    fake.error_message = f"Incorrect API key provided: {_KEY}"
    models = run.ServerModels(
        decider_model="policy", decider_base_url=fake.base_url, decider_api_key=_KEY
    )
    proc = subprocess.run(
        [sys.executable, "-P", str(_BENCH / "decider_probe.py")],
        env=_server_env(tmp_path, models),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    (line,) = proc.stdout.splitlines()
    assert json.loads(line)["outcome"] == "failed"
    assert _KEY not in proc.stdout
    assert _KEY not in proc.stderr


def test_a_resolve_only_run_records_the_decider_without_asking(
    fake: _FakeDecider, tmp_path: Path
) -> None:
    """A ``--stages answer judge`` resume needs the record, not a decider call."""
    models = run.ServerModels(
        decider_model="policy", decider_base_url=fake.base_url, decider_api_key=_KEY
    )
    probe = run._resolve_decider(_cfg(), _server_env(tmp_path, models), ask=False)
    assert probe is not None
    assert (probe.outcome, probe.model, probe.base_url) == (
        "resolved",
        "policy",
        fake.base_url,
    )
    assert fake.requests == []


def test_the_root_s_everos_toml_is_what_the_probe_reads(
    fake: _FakeDecider, tmp_path: Path
) -> None:
    """The server reads ``<root>/everos.toml``; so does the probe, at the same root."""
    (tmp_path / "everos.toml").write_text(
        f'[llm.decider]\nmodel = "from-toml"\nbase_url = "{fake.base_url}"\n'
        f'api_key = "{_KEY}"\n',
        encoding="utf-8",
    )
    probe = run._resolve_decider(
        _cfg(), _server_env(tmp_path, run.ServerModels()), ask=False
    )
    assert probe is not None
    assert (probe.model, probe.sources["model"]) == ("from-toml", "llm.decider")


def test_a_probe_that_crashes_stops_the_run_without_the_key(tmp_path: Path) -> None:
    """A setting EverOS rejects stops the probe before any report; the run stops too."""
    models = run.ServerModels(
        decider_model="policy", decider_base_url=_UNREACHABLE, decider_api_key=_KEY
    )
    env = _server_env(tmp_path, models, {"EVEROS_LLM__DECIDER__MAX_TOKENS": "0"})
    with pytest.raises(run.DeciderUnreachableError, match="without a report") as caught:
        run._resolve_decider(_cfg(), env, ask=True)
    assert _KEY not in str(caught.value)


def test_a_run_with_no_multiround_route_is_neither_probed_nor_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hybrid run makes no decider calls, so it has no endpoint to check.

    The gate is what the run does, not which fields are filled in: probing anyway
    would fail a hybrid run over an endpoint it never touches.
    """

    def _boom(*_a: Any, **_kw: Any) -> None:
        raise AssertionError("must not start a probe for a run with no decider")

    monkeypatch.setattr(run.subprocess, "run", _boom)
    assert run._resolve_decider(_cfg(methods="hybrid"), {}, ask=True) is None


def test_the_probe_runs_before_anything_is_recorded() -> None:
    """Checked at startup, not lazily: fail before the budget is spent, not after.

    And before the run record is built, which names the decider this resolves.
    """
    src = inspect.getsource(run._main_inner)
    probe_at = src.index(
        '_resolve_decider(config, server_env, ask="search" in args.stages)'
    )
    assert probe_at < src.index("_serving_from_env(config, _decider)")
    assert probe_at < src.index("_write_run_spec(")


# ---------------------------------------------------------------------------
# Reading the report
# ---------------------------------------------------------------------------


def test_a_thinking_failure_names_the_fix() -> None:
    """ "Thought instead of answering" has a specific fix; the message carries it."""
    probe = _probe(outcome="failed", reason="cut_short", error="generation cut short")
    with pytest.raises(run.DeciderUnreachableError, match="enable_thinking"):
        run._assert_decider_answers(probe)


def test_a_transport_failure_does_not_suggest_thinking() -> None:
    probe = _probe(outcome="failed", reason="transport", error="Connection error.")
    with pytest.raises(run.DeciderUnreachableError) as caught:
        run._assert_decider_answers(probe)
    assert "enable_thinking" not in str(caught.value)


def test_nothing_configured_says_so_instead_of_returning_quietly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Silence is what the probe exists to remove; it must not answer with silence."""
    run._assert_decider_answers(
        _probe(outcome="not_configured", error="LLM api_key and base_url missing")
    )
    assert "NOT PROBED" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The run record
# ---------------------------------------------------------------------------


def test_the_record_names_the_decider_the_servers_resolve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The decider comes from the probe; ``[llm.decider]`` is not the backbone's.

    ``EVEROS_LLM__`` is also the prefix of every ``EVEROS_LLM__<TASK>__*`` variable, so
    scanning it unfiltered filed the decider's settings under the backbone.
    """
    monkeypatch.setenv("EVEROS_LLM__MODEL", "backbone")
    monkeypatch.setenv("EVEROS_LLM__TIMEOUT_SECONDS", "120")
    monkeypatch.setenv("EVEROS_LLM__DECIDER__MODEL", "policy")
    monkeypatch.setenv("EVEROS_LLM__DECIDER__EXTRA", _NO_THINK)
    monkeypatch.setenv("EVEROS_LLM__BOUNDARY__MODEL", "cutter")
    probe = _probe(outcome="resolved")

    specs = {s.role: s for s in run._serving_from_env(_cfg(), probe)}

    decider = specs["decider"]
    assert (decider.model, decider.endpoint) == ("policy", probe.base_url)
    assert decider.extra["sources"] == probe.sources
    backbone = specs["backbone"]
    assert backbone.model == "backbone"
    assert backbone.extra == {"timeout_seconds": "120"}


def test_a_run_without_a_decider_records_none() -> None:
    roles = {s.role for s in run._serving_from_env(_cfg(methods="hybrid"), None)}
    assert "decider" not in roles
