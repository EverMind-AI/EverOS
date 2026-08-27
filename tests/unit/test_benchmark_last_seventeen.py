"""The last seventeen names in the migrated modules that no test referenced.

The partition of `benchmarks/` against the reference harnesses leaves three groups: 16
definitions copied byte-for-byte (verified by byte comparison), 30 rewritten (each
differentially tested against its own reference), and 104 written for this harness with
no counterpart. Seventeen names across those groups had no test referring to them at
all, and this file closes that: server lifecycle, the operator tooling, the CLI
dispatch, the vendor-inference helpers, and the re-extraction driver.

None of them is exempted. Where a real process is needed the process is stubbed, because
the alternative -- a sentence saying a live run covers it -- is what let six of these go
unchecked while claiming full coverage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

import run as run_mod  # noqa: E402
from adapters import base as ab  # noqa: E402
from adapters import evermembench, locomo, longmemeval, subtlememory  # noqa: E402

# ---------------------------------------------------------------------------
# Vendor inference: two static helpers the provider pin rests on
# ---------------------------------------------------------------------------

VENDORS = [
    ("openai/gpt-4.1-mini", "openai"),
    ("gpt-4o-mini", "openai"),
    ("anthropic/claude-sonnet-5", "anthropic"),
    ("claude-sonnet-5", "anthropic"),
    ("google/gemini-3.6-flash", "google"),
    ("gemini-3.6-flash", "google"),
    ("deepseek/deepseek-v4-pro", "deepseek"),
    ("qwen3-30b-a3b", ""),
    ("", ""),
]


@pytest.mark.parametrize(("model", "vendor"), VENDORS)
def test_vendor_of_places_every_shape_of_model_id(model: str, vendor: str) -> None:
    """A bare name that matches no vendor must return "" so it goes unpinned.

    Pinning it to a default vendor is how a bare qwen id ends up routed to a provider
    that does not serve it.
    """
    assert run_mod.LLMClientPool._vendor_of(model) == vendor


def test_model_override_is_inert_unset_and_both_forms_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_PROVIDER_JSON", raising=False)
    assert run_mod.LLMClientPool._model_override("openai/gpt-oss-120b") is None

    monkeypatch.setenv("OPENROUTER_PROVIDER_JSON", '{"only":["deepinfra"]}')
    assert run_mod.LLMClientPool._model_override("anything") == {"only": ["deepinfra"]}

    monkeypatch.setenv("OPENROUTER_PROVIDER_JSON", '{"gpt-oss":{"only":["novita"]}}')
    assert run_mod.LLMClientPool._model_override("openai/gpt-oss-120b") == {
        "only": ["novita"]
    }
    assert run_mod.LLMClientPool._model_override("google/gemini-3.6-flash") is None

    monkeypatch.setenv("OPENROUTER_PROVIDER_JSON", "{not json")
    assert run_mod.LLMClientPool._model_override("x") is None, "must degrade, not raise"


# ---------------------------------------------------------------------------
# Server lifecycle, with the process stubbed
# ---------------------------------------------------------------------------


class _FakeProc:
    """Enough of Popen for the fleet and the index gate."""

    def __init__(self, *a: Any, **kw: Any) -> None:
        self.env = kw.get("env") or {}
        self.pid = 4242
        self.args = a[0] if a else []
        self.returncode = None
        self._alive = True
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False

    def wait(self, timeout: float | None = None) -> int:
        self._alive = False
        return 0

    def communicate(self, *a: Any, **kw: Any) -> tuple[bytes, bytes]:
        self._alive = False
        return (b"", b"")

    def __enter__(self) -> _FakeProc:
        return self

    def __exit__(self, *_: Any) -> None:
        self.wait()


def test_the_fleet_starts_one_server_per_root_and_stops_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_ServerFleet.stop` was the half nothing referred to.

    A fleet that starts and never stops leaves a server holding the store, which is how
    an idle process sat on 70 GB of GPU for a day.
    """
    procs: list[_FakeProc] = []

    def _popen(*a: Any, **kw: Any) -> _FakeProc:
        p = _FakeProc(*a, **kw)
        procs.append(p)
        return p

    root = tmp_path / "store"
    root.mkdir()
    (root / "everos.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(run_mod.subprocess, "Popen", _popen)
    monkeypatch.setattr(
        run_mod.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0})()
    )
    fleet = run_mod._ServerFleet(1, root, first_port=0)
    monkeypatch.setattr(fleet, "_await_ready", lambda *a, **k: None)
    fleet.start()
    assert len(procs) == 1, f"started {len(procs)} servers for one root"
    assert fleet.urls and fleet.roots

    fleet.stop()
    assert procs[0].terminated or procs[0].killed, "the server was left running"


def test_wait_ready_refuses_a_store_with_no_cascade_database(tmp_path: Path) -> None:
    """It refuses rather than reporting ready, and the message names the path.

    Reporting ready on a store it cannot poll is the failure that matters here: ADD
    would then write its done marker over a store that never ingested anything.
    """
    with pytest.raises(RuntimeError, match="Cascade DB not found"):
        run_mod._wait_ready(
            str(tmp_path),
            0,
            "default",
            timeout_s=1,
            app_id="default",
            owner_id="nobody",
        )


def test_wait_ready_does_not_report_ready_for_an_owner_with_no_rows(
    tmp_path: Path,
) -> None:
    """The subtler case: the database exists but holds nothing for this owner.

    `_poll_cascade`'s SUM returns NULL there, and `None or 0` reads as "zero pending" --
    so a typo in the owner made the wait return instantly and ADD claim success. What it
    must not do is return a healthy-looking outcome after no work was ever observed.
    """
    import sqlite3

    db = tmp_path / ".index" / "sqlite"
    db.mkdir(parents=True)
    conn = sqlite3.connect(db / "system.db")
    conn.execute(
        "CREATE TABLE md_change_state (md_path TEXT, status TEXT, updated_at TEXT)"
    )
    conn.execute(
        "INSERT INTO md_change_state VALUES "
        "('/x/default/realowner/episodes/a.md', 'pending', '2026-01-01')"
    )
    conn.commit()
    conn.close()

    total, pending = run_mod._poll_cascade(db / "system.db", "%/realowner/%")
    assert (total, pending) == (1, 1), "the owner that has rows must be seen"
    total, pending = run_mod._poll_cascade(db / "system.db", "%/typo_owner/%")
    assert total == 0, "a non-matching owner must not report rows"
    assert pending == 0, (
        "SUM over no rows is NULL and must not read as work; the caller treats "
        "pending==0 as done"
    )


# ---------------------------------------------------------------------------
# Operator tooling: read-only, and it must not invent servers
# ---------------------------------------------------------------------------


def test_iter_servers_reports_nothing_when_no_server_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It walks /proc, so the check is a well-formed list, not a count."""
    rows = run_mod._iter_servers()
    assert isinstance(rows, list)
    for r in rows:
        assert len(r) == 3, r
        assert isinstance(r[0], int), r


def test_server_owner_and_version_degrade_on_a_dead_target() -> None:
    """Both are used while diagnosing; neither may raise on a dead pid or port."""
    assert run_mod._server_owner(999_999) is None
    v = run_mod._server_version("1")
    assert isinstance(v, str)


def test_print_servers_and_reap_servers_run_without_a_server(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--reap-servers` must not touch a server this run does not own."""
    monkeypatch.setattr(run_mod, "_iter_servers", lambda: [])
    run_mod._print_servers()
    assert capsys.readouterr().out != ""

    killed: list[int] = []
    monkeypatch.setattr(run_mod.os, "kill", lambda pid, sig: killed.append(pid))
    run_mod._reap_servers()
    assert killed == [], f"reaped {killed} with no servers listed"


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def test_main_dispatches_the_read_only_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    """`main` was rewritten from the reference's and nothing referred to it."""
    seen: list[str] = []
    monkeypatch.setattr(run_mod, "_print_servers", lambda: seen.append("list"))
    monkeypatch.setattr(sys, "argv", ["run.py", "--list-servers"])
    with pytest.raises(SystemExit) as e:
        run_mod.main()
    assert e.value.code in (0, None), e.value.code
    assert seen == ["list"]


def test_main_refuses_an_unknown_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo produced a raw FileNotFoundError traceback naming a .toml path.

    The missing-name case already printed the four available benchmarks; the wrong-name
    case did not, which is the case an operator actually hits.
    """
    monkeypatch.setattr(sys, "argv", ["run.py", "not_a_benchmark"])
    with pytest.raises(SystemExit) as e:
        run_mod.main()
    assert e.value.code != 0


def test_main_inner_is_what_main_delegates_to() -> None:
    """Kept separate so the dispatch can be tested without the run."""
    import inspect

    assert "_main_inner" in inspect.getsource(run_mod.main)


# ---------------------------------------------------------------------------
# The re-extraction driver
# ---------------------------------------------------------------------------


def test_judge_spec_is_declared_by_every_adapter_and_records_provenance() -> None:
    """It is provenance only, but every adapter must be able to state it."""
    for mod in (locomo, longmemeval, subtlememory, evermembench):
        spec = mod.judge_spec()
        assert isinstance(spec, dict), f"{mod.name}: {type(spec)}"
        assert spec, f"{mod.name}: empty judge spec"
    assert hasattr(ab.DatasetAdapter, "judge_spec")


# ---------------------------------------------------------------------------
# The stage orchestrator
# ---------------------------------------------------------------------------


def test_run_conversation_drives_search_answer_and_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one name in the migrated modules no test referred to.

    It is what turns a conversation into three artefacts, so the check is that it
    produces all three, in question order, from one stubbed client each -- not that it
    merely returns. ADD is left out: it is the stage that cannot be replayed cheaply,
    and it is covered separately by the payload-stream differential.
    """
    from config import BenchmarkConfig

    cfg = BenchmarkConfig.from_toml("locomo")
    if not Path(cfg.data_path).exists():
        pytest.skip("dataset not present")

    class _Client:
        """Serves the search endpoint and records the questions it was asked."""

        def __init__(self) -> None:
            self.asked: list[str] = []

        def post(self, path: str, data: dict, quiet: bool = False) -> tuple[int, dict]:
            self.asked.append(str(data.get("query")))
            return 200, {
                "data": {
                    "episodes": [
                        {
                            "id": "ep_1",
                            "content": "a narrative",
                            "summary": "a summary",
                            "session_id": "locomo_conv0_s1",
                            "score": 1.0,
                        }
                    ],
                    "profiles": [],
                }
            }

    class _LLM:
        def __init__(self, reply: str) -> None:
            self._reply = reply
            self.chat = self
            self.completions = self
            self.calls = 0

        def create(self, **kwargs: Any) -> Any:
            self.calls += 1
            msg = type("M", (), {"content": self._reply})()
            usage = type(
                "U", (), {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
            )()
            return type(
                "R",
                (),
                {"choices": [type("C", (), {"message": msg})()], "usage": usage},
            )()

    client = _Client()
    answer_llm = _LLM("3 May 2023")
    judge_llm = _LLM('{"label": "CORRECT"}')
    monkeypatch.setattr(run_mod, "EverosClient", lambda **_: client)
    # Two questions is enough to prove ordering and per-question artefacts.
    monkeypatch.setattr(run_mod, "_stratified_sample", lambda qa, n: qa[:2])

    args = type(
        "A",
        (),
        {
            "base_url": ["http://127.0.0.1:1"],
            "everos_root": [str(tmp_path / "store")],
            "app_id": "",
            "project_id": "",
            "methods": "hybrid",
            "smoke": True,
            # `--max-messages` overrides --smoke's hardcoded 50; 0 keeps --smoke.
            "max_messages": 0,
            "reuse_store": False,
            "results_root": str(tmp_path),
            "run_name": "r",
            "data_path": cfg.data_path,
            "answer_model": "",
            "judge_model": "",
            "decider_model": "",
            "backbone_model": "",
            "backbone_base_url": "",
            "conv": [0],
            "servers": 0,
            "stages": ["search", "answer", "judge"],
            "benchmark_name": "locomo",
            "config": None,
        },
    )()

    run_mod.run_conversation(
        0,
        args=args,
        config=cfg,
        stages=["search", "answer", "judge"],
        answer_client=answer_llm,
        judge_client=judge_llm,
        data_path=cfg.data_path,
        output_dir=tmp_path / "out",
    )

    conv = tmp_path / "out" / "conv0"
    # The artefact is named after the method the CONFIG selects, not `args.methods`:
    # `run_conversation` reads `config.methods`, so a run overriding the method on the
    # command line still writes files named by the config's. Asserted rather than
    # assumed, because the report reads these names back.
    method = cfg.methods.split(",")[0].strip()
    rows = {}
    for stage in ("search", "answer", "judge"):
        f = conv / f"{stage}_{method}.jsonl"
        assert f.exists(), (
            f"{stage} produced no artefact; conv0 holds "
            f"{sorted(p.name for p in conv.iterdir())}"
        )
        rows[stage] = [
            json.loads(ln) for ln in f.read_text(encoding="utf-8").splitlines() if ln
        ]
    assert len({len(v) for v in rows.values()}) == 1, (
        f"the stages disagree on how many questions there were: "
        f"{ {k: len(v) for k, v in rows.items()} }"
    )
    assert rows["search"], "no questions were run"
    for stage, got in rows.items():
        assert [r["index"] for r in got] == sorted(r["index"] for r in got), (
            f"{stage} is not in question order"
        )
    assert client.asked, "search never reached the client"
    assert answer_llm.calls == len(rows["answer"])
    assert all(r["generated_answer"] == "3 May 2023" for r in rows["answer"])
    assert all(r["is_correct"] for r in rows["judge"])
