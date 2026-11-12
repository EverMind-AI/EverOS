"""The first-run path through benchmarks/: every gap stops the run before it spends.

Pins the contracts a new user (or a coding agent following the README) relies on:
the download script lands each dataset where its config reads it, the preflight lists
every missing dataset and key before a server starts -- asking only for the providers
the retrieval methods use, so ``keyword`` runs on one key -- the fleet refuses to ADD
when a server lacks a capability those methods need, ``run.py`` reads
``benchmarks/.env`` before it expands the config, a smoke report says so on its front
page, and ``--smoke`` means one conversation, through ``run.py`` and ``reproduce.sh``.
"""

from __future__ import annotations

import inspect
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import preflight  # noqa: E402
import run as run_mod  # noqa: E402
from adapters import subtlememory  # noqa: E402
from config import BenchmarkConfig  # noqa: E402

_ALL_KEYS = {
    "EVEROS_LLM__API_KEY": "k",
    "EVEROS_EMBEDDING__API_KEY": "k",
    "ANSWER_API_KEY": "k",
    "JUDGE_API_KEY": "k",
}
_NEEDS_BASH = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="the shell entry points are bash scripts",
)


# ---------------------------------------------------------------------------
# preflight: data and keys, before any server starts
# ---------------------------------------------------------------------------


def _problems(env: dict[str, str], data: Path, **kw: Any) -> list[str]:
    opts: dict[str, Any] = {
        "benchmark": "locomo",
        "data_path": str(data),
        "stages": ["add", "search", "answer", "judge"],
        "methods": ["llm_multiround"],
        "starts_servers": True,
        **kw,
    }
    return preflight.blocking_problems(env, **opts)


@pytest.mark.parametrize(
    ("methods", "needs"),
    [
        (["keyword"], []),
        (["llm_multiround"], ["embed"]),
        (["vector"], ["embed"]),
        (["hybrid"], ["embed"]),
        (["agentic"], ["embed", "rerank"]),
        (["keyword", "agentic"], ["embed", "rerank"]),
    ],
)
def test_capabilities_follow_the_retrieval_method(
    methods: list[str], needs: list[str]
) -> None:
    """The same rule the EverOS search manager enforces with HTTP 422."""
    assert preflight.required_capabilities(methods) == needs


def test_keyword_runs_on_the_llm_key_alone(tmp_path: Path) -> None:
    """EverOS Tier 1: one OpenRouter key, no embedding, no rerank."""
    data = tmp_path / "d.json"
    data.write_text("[]")
    env = {k: v for k, v in _ALL_KEYS.items() if k != "EVEROS_EMBEDDING__API_KEY"}
    assert _problems(env, data, methods=["keyword"]) == []


def test_agentic_asks_for_a_rerank_key_before_add_spends(tmp_path: Path) -> None:
    """LoCoMo's shipped method; without the key every search fails with HTTP 422."""
    data = tmp_path / "d.json"
    data.write_text("[]")
    [problem] = _problems(_ALL_KEYS, data, methods=["agentic"])
    assert problem.startswith("EVEROS_RERANK__API_KEY is not set")


def test_a_complete_setup_has_no_problems(tmp_path: Path) -> None:
    data = tmp_path / "locomo10.json"
    data.write_text("[]")
    assert _problems(_ALL_KEYS, data) == []


def test_every_gap_is_reported_at_once(tmp_path: Path) -> None:
    """A first run is told everything it is missing, not one item per attempt."""
    problems = _problems({}, tmp_path / "missing.json")
    text = "\n".join(problems)
    assert "bash benchmarks/download_data.sh locomo" in text
    for key in _ALL_KEYS:
        assert key in text, f"{key} not reported"
    assert len(problems) == 5


def test_the_old_placeholder_counts_as_unset(tmp_path: Path) -> None:
    """An older .env.example shipped `sk-...`, which is non-empty but not a key."""
    data = tmp_path / "d.json"
    data.write_text("[]")
    env = {**_ALL_KEYS, "ANSWER_API_KEY": "sk-..."}
    [problem] = _problems(env, data)
    assert problem.startswith("ANSWER_API_KEY is not set -- ")


def test_server_keys_are_not_required_of_servers_the_caller_manages(
    tmp_path: Path,
) -> None:
    data = tmp_path / "d.json"
    data.write_text("[]")
    env = {"ANSWER_API_KEY": "k", "JUDGE_API_KEY": "k"}
    assert _problems(env, data, starts_servers=False) == []


def test_a_started_server_needs_its_llm_key_even_when_it_only_hosts(
    tmp_path: Path,
) -> None:
    """EverOS refuses to start without [llm]; embedding is only needed for work."""
    data = tmp_path / "d.json"
    data.write_text("[]")
    env = {"ANSWER_API_KEY": "k", "JUDGE_API_KEY": "k"}
    [problem] = _problems(env, data, stages=["answer", "judge"])
    assert problem.startswith("EVEROS_LLM__API_KEY is not set")


def test_provider_keys_are_read_from_what_the_servers_inherit(tmp_path: Path) -> None:
    """A key the config's retrieval_env hands the servers counts, not just .env's."""
    data = tmp_path / "d.json"
    data.write_text("[]")
    env = {"ANSWER_API_KEY": "k", "JUDGE_API_KEY": "k"}
    server_env = {"EVEROS_LLM__API_KEY": "k", "EVEROS_EMBEDDING__API_KEY": "k"}
    assert _problems(env, data, server_env=server_env) == []
    unexpanded = {**server_env, "EVEROS_EMBEDDING__API_KEY": "${EMBED_KEY}"}
    [problem] = _problems(env, data, server_env=unexpanded)
    assert problem.startswith("EVEROS_EMBEDDING__API_KEY is not set")


def test_the_preflight_reads_the_environment_the_fleet_launches_with() -> None:
    """The servers' keys are layered by server_launch_env, as for the fleet itself."""
    src = inspect.getsource(run_mod.main)
    preflight_at = src.index("blocking_problems(")
    assert "server_env=server_launch_env(" in src[preflight_at:]
    assert preflight_at < src.index("fleet.start()")


def test_the_blocked_message_says_how_to_create_the_env_file(tmp_path: Path) -> None:
    message = preflight.format_problems(["x"], env_file=tmp_path / ".env")
    assert message.startswith("Not started -- nothing has been spent yet.")
    assert "cp benchmarks/.env.example benchmarks/.env" in message


def _argv(tmp_path: Path, data: Path, *extra: str) -> list[str]:
    return [
        "run.py",
        "locomo",
        "--conv",
        "0",
        "--smoke",
        "--data-path",
        str(data),
        "--results-root",
        str(tmp_path / "results"),
        "--everos-root",
        str(tmp_path / "store"),
        *extra,
    ]


def test_main_stops_before_the_fleet_when_data_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_fleet(self: Any) -> None:
        raise AssertionError("the fleet started before the preflight blocked")

    monkeypatch.setattr(run_mod._ServerFleet, "start", _no_fleet)
    # A developer's own benchmarks/.env must not leak keys into this process.
    monkeypatch.setattr(run_mod, "load_dotenv", lambda *_a, **_k: False)
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, tmp_path / "missing.json"))
    with pytest.raises(SystemExit) as exc:
        run_mod.main()
    assert "bash benchmarks/download_data.sh locomo" in str(exc.value.code)


def test_a_passing_preflight_says_so_before_the_fleet_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _FleetStartedError(Exception):
        pass

    def _stop_at_fleet(self: Any) -> None:
        raise _FleetStartedError

    data = tmp_path / "locomo10.json"
    data.write_text("[]")
    monkeypatch.setattr(run_mod._ServerFleet, "start", _stop_at_fleet)
    monkeypatch.setattr(run_mod, "load_dotenv", lambda *_a, **_k: False)
    for key in ("EVEROS_LLM__API_KEY", "ANSWER_API_KEY", "JUDGE_API_KEY"):
        monkeypatch.setenv(key, "k")
    monkeypatch.setattr(sys, "argv", _argv(tmp_path, data, "--methods", "keyword"))
    with pytest.raises(_FleetStartedError):
        run_mod.main()
    assert "Preflight: data and keys ready" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# fleet: the capabilities the methods need must be there before ADD
# ---------------------------------------------------------------------------


def test_servers_lacking_reads_the_health_capability() -> None:
    health = {
        "http://a": {"capabilities": {"embed": True, "rerank": False}},
        "http://b": {"capabilities": {"embed": False, "rerank": True}},
        "http://c": {},
    }
    assert preflight.servers_lacking(health, "embed") == ["http://b", "http://c"]
    assert preflight.servers_lacking(health, "rerank") == ["http://a", "http://c"]


class _Resp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._body = json.dumps(payload).encode()

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def _fleet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capabilities: dict[str, bool]
) -> Any:
    fleet = run_mod._ServerFleet(1, tmp_path / "store", first_port=19999)
    fleet.urls = ["http://127.0.0.1:19999"]
    fleet.logs = [tmp_path / "store" / "server.log"]
    monkeypatch.setattr(
        run_mod.urllib.request,
        "urlopen",
        lambda url, timeout: _Resp({"capabilities": capabilities}),
    )
    return fleet


def test_the_fleet_goes_on_when_every_capability_is_there(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fleet = _fleet(tmp_path, monkeypatch, {"embed": True, "rerank": True})
    fleet.assert_capabilities(["embed", "rerank"])
    out = capsys.readouterr().out
    assert "Embedding answered on 1 server(s)" in out
    assert "Rerank configured on 1 server(s)" in out


@pytest.mark.parametrize(
    ("capability", "problem"),
    [
        ("embed", "embedding service did not answer"),
        ("rerank", "No rerank provider is configured"),
    ],
)
def test_the_fleet_refuses_to_add_without_a_needed_capability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capability: str, problem: str
) -> None:
    fleet = _fleet(tmp_path, monkeypatch, {"embed": True, capability: False})
    with pytest.raises(SystemExit) as exc:
        fleet.assert_capabilities([capability])
    message = str(exc.value.code)
    assert problem in message
    assert str(tmp_path / "store" / "server.log") in message


def test_a_keyword_run_asks_the_fleet_for_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fleet = _fleet(tmp_path, monkeypatch, {"embed": False, "rerank": False})
    fleet.assert_capabilities(preflight.required_capabilities(["keyword"]))


def test_a_server_that_stops_answering_names_its_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fleet = _fleet(tmp_path, monkeypatch, {})

    def _refused(url: str, timeout: float) -> Any:
        raise run_mod.urllib.error.URLError("connection refused")

    monkeypatch.setattr(run_mod.urllib.request, "urlopen", _refused)
    with pytest.raises(SystemExit) as exc:
        fleet.assert_capabilities(["embed"])
    message = str(exc.value.code)
    assert "stopped answering /health" in message
    assert str(tmp_path / "store" / "server.log") in message


def test_the_capability_check_runs_before_any_stage_does() -> None:
    src = inspect.getsource(run_mod.main)
    check_at = src.index("fleet.assert_capabilities(")
    assert check_at < src.index("_main_inner(args, config, server_env)")


# ---------------------------------------------------------------------------
# report.txt: a smoke score must not read like a result
# ---------------------------------------------------------------------------


def _report(tmp_path: Path, run_spec: dict[str, Any], config: BenchmarkConfig) -> str:
    path = tmp_path / "report.txt"
    run_mod._write_report_txt(path, {}, [0], config, run_spec, "")
    return path.read_text()


def test_a_smoke_report_says_so_on_its_front_page(tmp_path: Path) -> None:
    text = _report(tmp_path, {"smoke": True, "max_messages": 50}, BenchmarkConfig())
    assert "Scope:          SMOKE -- first 50 messages" in text
    assert "not comparable to published results" in text


def test_a_trimmed_report_says_so_too(tmp_path: Path) -> None:
    text = _report(tmp_path, {"max_messages": 400}, BenchmarkConfig())
    assert "Scope:          first 400 messages per conversation" in text


def test_a_full_report_carries_no_scope_warning(tmp_path: Path) -> None:
    assert "Scope:" not in _report(tmp_path, {}, BenchmarkConfig())


def _decider_spec(source: str) -> dict[str, Any]:
    """A run_spec.json serving list as `_serving_from_env` records the decider."""
    return {
        "serving": [
            {"role": "answer", "model": "a"},
            {
                "role": "decider",
                "model": "qwen3.6-27B",
                "endpoint": "http://h/v1",
                "extra": {"sources": {"model": source, "base_url": source}},
            },
        ]
    }


def test_the_report_names_the_decider_the_run_recorded(tmp_path: Path) -> None:
    config = BenchmarkConfig(methods="llm_multiround")
    text = _report(tmp_path, _decider_spec("llm.decider"), config)
    assert (
        "Decider model:  qwen3.6-27B @ http://h/v1 (model from [llm.decider])\n" in text
    )


def test_the_report_names_a_decider_that_is_the_llm_model(tmp_path: Path) -> None:
    config = BenchmarkConfig(methods="llm_multiround")
    text = _report(tmp_path, _decider_spec("llm"), config)
    assert "-- the [llm] model; no separate decider was configured" in text


def test_a_report_without_a_recorded_decider_says_so(tmp_path: Path) -> None:
    text = _report(tmp_path, {}, BenchmarkConfig(methods="llm_multiround"))
    assert "Decider model:  not recorded in run_spec.json\n" in text


def test_a_report_of_a_run_without_a_decider_says_none_was_used(
    tmp_path: Path,
) -> None:
    text = _report(tmp_path, {}, BenchmarkConfig(methods="agentic"))
    assert "Decider model:  not used by agentic\n" in text


# ---------------------------------------------------------------------------
# parse_args: what a direct run.py invocation resolves
# ---------------------------------------------------------------------------


def _parse(monkeypatch: pytest.MonkeyPatch, *argv: str, dotenv: Any = None) -> Any:
    """parse_args on `argv`, with `dotenv` standing in for loading benchmarks/.env."""
    for var in ("BENCH_DECIDER_MODEL", "BENCH_DECIDER_BASE_URL", "BENCH_DATA_LOCOMO"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(run_mod, "load_dotenv", dotenv or (lambda *_a, **_k: False))
    monkeypatch.setattr(sys, "argv", ["run.py", "locomo", *argv])
    return run_mod.parse_args()


def test_the_methods_override_is_what_the_run_resolves(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Applied with the config, so the decider notice and every later read see it."""
    _args, config = _parse(monkeypatch, "--smoke", "--methods", "llm_multiround")
    assert config.methods == "llm_multiround"
    assert "decider  no endpoint" in capsys.readouterr().out


def test_a_misspelt_method_on_the_command_line_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit):
        _parse(monkeypatch, "--smoke", "--methods", "keywrod")


def test_smoke_alone_means_conversation_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    args, _config = _parse(monkeypatch, "--smoke")
    assert args.conv == [0]


def test_benchmarks_env_is_loaded_before_the_config_expands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run directly, run.py has no reproduce.sh to have sourced .env first."""
    data = tmp_path / "elsewhere.json"

    def _dotenv(*_a: Any, **_k: Any) -> bool:
        monkeypatch.setenv("BENCH_DATA_LOCOMO", str(data))
        return True

    args, _config = _parse(monkeypatch, "--smoke", dotenv=_dotenv)
    assert args.data_path == str(data)


@pytest.mark.parametrize(
    ("servers", "names"),
    [(1, ["store"]), (3, ["store_s0", "store_s1", "store_s2"])],
)
def test_every_store_lives_inside_the_run_directory(
    tmp_path: Path, servers: int, names: list[str]
) -> None:
    """Deleting the run directory removes every store a fleet built."""
    roots = run_mod._store_roots(tmp_path / "run" / "store", servers)
    assert [r.name for r in roots] == names
    assert all(r.parent == tmp_path / "run" for r in roots)


def test_a_stage_only_rerun_finds_a_fleets_stores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a multi-server ADD there is no `store`, only `store_s<i>`."""
    (tmp_path / "LoCoMo" / "r" / "store_s0").mkdir(parents=True)
    args, _config = _parse(
        monkeypatch,
        "--run-name",
        "r",
        "--results-root",
        str(tmp_path / "LoCoMo"),
        "--stages",
        "search",
        "answer",
        "judge",
    )
    assert args.everos_root == [str(tmp_path / "LoCoMo" / "r" / "store")]


def test_the_run_prints_the_stores_it_uses(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    roots = run_mod._store_roots(tmp_path / "store", 3)
    run_mod._print_stores(roots, reusing=True)
    out = capsys.readouterr().out
    assert out.startswith(f"store    reusing {tmp_path / 'store_s0'} .. store_s2")
    assert "(3 stores, one per server)" in out


# ---------------------------------------------------------------------------
# SubtleMemory: one level off is an error that names the fix
# ---------------------------------------------------------------------------


def test_subtlememory_nested_one_level_deep_names_the_download_script(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "data" / "subtlememory" / "persona_0"
    nested.mkdir(parents=True)
    (nested / "bench_instances.json").write_text("[]")
    (nested / "history_sessions.json").write_text("[]")
    with pytest.raises(FileNotFoundError, match=r"download_data\.sh subtlememory"):
        subtlememory.load_units(str(tmp_path))


# ---------------------------------------------------------------------------
# shell entry points
# ---------------------------------------------------------------------------


def _executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _sandbox(tmp_path: Path, script: str) -> Path:
    """A copy of the script in a tree of its own: both scripts cd to their parent."""
    (tmp_path / "benchmarks").mkdir(exist_ok=True)
    shutil.copy(BENCH / script, tmp_path / "benchmarks" / script)
    return tmp_path / "benchmarks" / script


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    keep = {k: os.environ[k] for k in ("HOME", "TMPDIR") if k in os.environ}
    return {
        **keep,
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
        **extra,
    }


_STUB_CURL = """#!/usr/bin/env bash
# Writes the URL it was asked for into the -o destination; logs every call.
while [ $# -gt 0 ]; do
  case "$1" in
    -o) dest=$2; shift 2 ;;
    http*) url=$1; shift ;;
    *) shift ;;
  esac
done
echo "$url" >> "$(dirname "$0")/curl.log"
printf '%s' "$url" > "$dest"
"""


@_NEEDS_BASH
def test_download_lands_each_file_where_the_adapter_reads_it(tmp_path: Path) -> None:
    script = _sandbox(tmp_path, "download_data.sh")
    _executable(tmp_path / "bin" / "curl", _STUB_CURL)
    _executable(
        tmp_path / ".venv" / "bin" / "python",
        '#!/usr/bin/env bash\necho "$@" > "$(dirname "$0")/converted"\n',
    )
    env = _env(tmp_path)
    for target in ("locomo", "longmemeval", "evermembench", "subtlememory"):
        subprocess.run(["bash", str(script), target], env=env, check=True)

    data = tmp_path / "benchmarks" / "data"
    assert (data / "locomo10.json").read_text().endswith("/data/locomo10.json")
    assert (data / "locomo-audit-errors.json").read_text().endswith("/errors.json")
    # Upstream has no extension; the config reads `.json`.
    assert (data / "longmemeval_s.json").read_text().endswith("/main/longmemeval_s")
    raw = data / "raw" / "EverMemBench-Dynamic"
    assert sorted(p.relative_to(raw).as_posix() for p in raw.rglob("*.json")) == [
        f"{t}/{f}"
        for t in ("01", "02", "03", "04", "05")
        for f in ("dialogue.json", f"qa_{t}.json")
    ]
    converted = tmp_path / ".venv" / "bin" / "converted"
    assert "benchmarks.adapters.evermembench import main" in converted.read_text()
    personas = data / "subtlememory"
    assert len(list(personas.glob("persona_*/bench_instances.json"))) == 10
    assert len(list(personas.glob("persona_*/history_sessions.json"))) == 10
    assert not list(data.rglob("*.part"))


@_NEEDS_BASH
def test_download_again_keeps_what_is_there(tmp_path: Path) -> None:
    script = _sandbox(tmp_path, "download_data.sh")
    _executable(tmp_path / "bin" / "curl", _STUB_CURL)
    env = _env(tmp_path)
    subprocess.run(["bash", str(script), "subtlememory"], env=env, check=True)
    log = tmp_path / "bin" / "curl.log"
    first = log.read_text().count("\n")
    out = subprocess.run(
        ["bash", str(script), "subtlememory"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert log.read_text().count("\n") == first == 20
    assert out.count("  have  ") == 20


@_NEEDS_BASH
def test_download_names_the_choices_for_an_unknown_benchmark(tmp_path: Path) -> None:
    script = _sandbox(tmp_path, "download_data.sh")
    proc = subprocess.run(
        ["bash", str(script), "nope"],
        env=_env(tmp_path),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "locomo|longmemeval|evermembench|subtlememory|all" in proc.stderr


_STUB_PYTHON = """#!/usr/bin/env bash
# Stands in for both the banner and run.py: prints its argv.
echo "ARGV $*"
"""


def _reproduce(
    tmp_path: Path, *args: str, **env: str
) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "benchmarks" / "reproduce.sh"
    if not script.exists():
        _sandbox(tmp_path, "reproduce.sh")
    return subprocess.run(
        ["bash", str(script), *args],
        env=_env(tmp_path, **env),
        capture_output=True,
        text=True,
    )


@_NEEDS_BASH
def test_reproduce_explains_a_missing_env_file_and_venv(tmp_path: Path) -> None:
    proc = _reproduce(tmp_path, "--smoke")
    assert proc.returncode == 1
    assert "cp benchmarks/.env.example benchmarks/.env" in proc.stderr
    assert "REQUIRED" in proc.stderr

    (tmp_path / "benchmarks" / ".env").write_text("")
    proc = _reproduce(tmp_path, "--smoke")
    assert proc.returncode == 1
    assert "uv sync" in proc.stderr


@_NEEDS_BASH
@pytest.mark.parametrize(
    ("args", "env", "expected"),
    [
        # run.py owns the smoke default, so no --conv is passed for it.
        (("--smoke",), {}, "--run-name locomo_smoke --stages"),
        ((), {}, "--run-name locomo --conv all --stages"),
        (
            ("--smoke",),
            {"CONV": "3"},
            "--run-name locomo_conv3_smoke --conv 3 --stages",
        ),
        ((), {"CONV": "0 1"}, "--run-name locomo_conv0-1 --conv 0 1 --stages"),
    ],
)
def test_reproduce_names_and_scopes_the_run(
    tmp_path: Path, args: tuple[str, ...], env: dict[str, str], expected: str
) -> None:
    _sandbox(tmp_path, "reproduce.sh")
    (tmp_path / "benchmarks" / ".env").write_text("")
    _executable(tmp_path / ".venv" / "bin" / "python", _STUB_PYTHON)
    proc = _reproduce(tmp_path, *args, **env)
    assert proc.returncode == 0, proc.stderr
    argv = next(
        line for line in proc.stdout.splitlines() if "benchmarks/run.py" in line
    )
    assert expected in argv
