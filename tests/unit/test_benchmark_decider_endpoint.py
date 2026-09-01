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

Two things are pinned here: the endpoint must be settable alongside the model, and a
decider that does not answer must raise before the first question.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

run = importlib.import_module("run")


_NO_THINK = '{"extra_body": {"chat_template_kwargs": {"enable_thinking": false}}}'


def _cfg(**over: Any) -> Any:
    """A config carrying only the decider fields this module reasons about."""
    base: dict[str, Any] = {
        "decider_model": "policy",
        "decider_base_url": "http://127.0.0.1:9360/v1",
        "decider_api_key": "EMPTY",
        "backbone_model": "",
        "backbone_base_url": "",
        "backbone_api_key": "",
        "retrieval_env": {},
        "parsed_methods": ["llm_multiround"],
    }
    base.update(over)
    return type("Cfg", (), base)()


class _Resp:
    def __init__(self, content: str | None, reasoning: str | None = None) -> None:
        msg = type("M", (), {"content": content, "reasoning_content": reasoning})()
        self.choices = [type("C", (), {"message": msg})()]


class _Client:
    """Stands in for openai.OpenAI, recording where the call was actually addressed."""

    seen: dict[str, Any] = {}

    def __init__(self, **kw: Any) -> None:
        _Client.seen = dict(kw)
        outer = self

        class _Completions:
            def create(self, **ckw: Any) -> _Resp:
                _Client.seen.update(ckw)
                return outer._reply()

        self.chat = type("Chat", (), {"completions": _Completions()})()

    def _reply(self) -> _Resp:
        raise NotImplementedError


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


def test_live_decider_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """A decider returning content is accepted, and is addressed at its own endpoint."""

    class Ok(_Client):
        def _reply(self) -> _Resp:
            return _Resp("ready")

    monkeypatch.setattr(run.openai, "OpenAI", Ok)
    run._assert_decider_answers(_cfg())
    assert Ok.seen["base_url"] == "http://127.0.0.1:9360/v1"
    assert Ok.seen["model"] == "policy"


def test_unreachable_decider_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404 for the model name must stop the run.

    This is the exact production failure: the endpoint is up and healthy, and refuses
    only this model.
    """

    class NotFound(_Client):
        def _reply(self) -> _Resp:
            raise RuntimeError("Error code: 404 - model 'policy' not found")

    monkeypatch.setattr(run.openai, "OpenAI", NotFound)
    with pytest.raises(run.DeciderUnreachableError, match="did not answer"):
        run._assert_decider_answers(_cfg())


def test_empty_completion_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A reasoning decider that spends its budget thinking returns nothing usable.

    The decider's own parser treats an empty reply as a failed attempt, so this reaches
    the same fallback by a different route and has to be caught the same way.
    """

    class Empty(_Client):
        def _reply(self) -> _Resp:
            return _Resp("")

    monkeypatch.setattr(run.openai, "OpenAI", Empty)
    with pytest.raises(run.DeciderUnreachableError, match="empty completion"):
        run._assert_decider_answers(_cfg())


def test_a_run_with_no_multiround_route_is_not_probed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hybrid run makes no decider calls, so it has no endpoint to check.

    The gate is what the run does, not which fields are filled in: probing anyway
    would fail a hybrid run over an endpoint it never touches.
    """

    def _boom(**_kw: Any) -> None:
        raise AssertionError("must not construct a client for a run with no decider")

    monkeypatch.setattr(run.openai, "OpenAI", _boom)
    run._assert_decider_answers(_cfg(parsed_methods=["hybrid"]))


def test_an_inherited_endpoint_is_still_probed(monkeypatch: pytest.MonkeyPatch) -> None:
    """`[decider].model` with no decider URL is a valid runtime config, and was skipped.

    `get_decider_llm_client()` falls back to the backbone's endpoint field by field, so
    this shape runs -- and when that endpoint does not serve the named model, every
    decider call fails inside the benchmark instead of before its first question. The
    old gate required both fields to be set explicitly and returned silently otherwise,
    which left the one case the probe exists for uncovered.
    """

    class Ok(_Client):
        def _reply(self) -> _Resp:
            return _Resp("ready")

    monkeypatch.setattr(run.openai, "OpenAI", Ok)
    run._assert_decider_answers(
        _cfg(
            decider_base_url="",
            decider_api_key="",
            backbone_base_url="http://backbone.invalid/v1",
            backbone_api_key="bk",
        )
    )
    assert Ok.seen["base_url"] == "http://backbone.invalid/v1"
    assert Ok.seen["api_key"] == "bk"
    assert Ok.seen["model"] == "policy", "the decider's own model, not the backbone's"


def test_a_fully_inherited_decider_is_probed_as_the_backbone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No `[decider]` at all: the backbone decides, and nothing else probes it.

    `_assert_decider_answers` is the only probe in this harness -- the comment that
    sent this case away said the backbone had "its own probe", and it has none.
    """

    class Ok(_Client):
        def _reply(self) -> _Resp:
            return _Resp("ready")

    monkeypatch.setattr(run.openai, "OpenAI", Ok)
    run._assert_decider_answers(
        _cfg(
            decider_model="",
            decider_base_url="",
            decider_api_key="",
            backbone_model="backbone",
            backbone_base_url="http://backbone.invalid/v1",
        )
    )
    assert Ok.seen["model"] == "backbone"
    assert Ok.seen["base_url"] == "http://backbone.invalid/v1"


def test_nothing_resolvable_says_so_instead_of_returning_quietly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Silence is what the probe exists to remove; it must not answer with silence."""

    def _boom(**_kw: Any) -> None:
        raise AssertionError("nothing to probe, so nothing should be constructed")

    monkeypatch.setattr(run.openai, "OpenAI", _boom)
    monkeypatch.delenv("EVEROS_LLM__BASE_URL", raising=False)
    monkeypatch.delenv("EVEROS_LLM__MODEL", raising=False)
    run._assert_decider_answers(
        _cfg(decider_model="", decider_base_url="", decider_api_key="")
    )
    assert "NOT PROBED" in capsys.readouterr().out


def test_the_probe_sends_the_decider_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real call is capped now, so an uncapped probe vouches for a different call.

    `LLMRoundDecider.__call__` sends `[decider].max_tokens`. A probe that omits it
    would pass on a budget the run never gets.
    """

    class Ok(_Client):
        def _reply(self) -> _Resp:
            return _Resp("ready")

    monkeypatch.setattr(run.openai, "OpenAI", Ok)
    monkeypatch.delenv("EVEROS_DECIDER__MAX_TOKENS", raising=False)
    run._assert_decider_answers(_cfg())
    assert Ok.seen["max_tokens"] == run._DECIDER_MAX_TOKENS_DEFAULT

    monkeypatch.setattr(run.openai, "OpenAI", Ok)
    run._assert_decider_answers(
        _cfg(retrieval_env={"EVEROS_DECIDER__MAX_TOKENS": "64"})
    )
    assert Ok.seen["max_tokens"] == 64


def test_probe_runs_before_the_search_stage() -> None:
    """Checked at startup, not lazily: fail before the budget is spent, not after."""
    src = (_BENCH / "run.py").read_text(encoding="utf-8")
    assert 'if "search" in args.stages:\n        _assert_decider_answers(config)' in src


def test_the_probe_sends_the_decider_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe must make the request the decider makes, not a simpler one.

    A re-run failed here: the endpoint was finally correct, but the
    probe omitted `enable_thinking=false` while the servers had it, so a qwen
    checkpoint served with `--reasoning-parser` spent the probe's whole budget in
    `reasoning_content` and returned empty `content`. A probe that fails runs which
    would have worked gets switched off, and then it protects nothing.
    """

    class Ok(_Client):
        def _reply(self) -> _Resp:
            return _Resp("ready")

    monkeypatch.setattr(run.openai, "OpenAI", Ok)
    run._assert_decider_answers(
        _cfg(retrieval_env={"EVEROS_DECIDER__EXTRA": _NO_THINK})
    )
    assert Ok.seen["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_the_extra_is_also_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launch scripts export it; configs declare it. The servers read either."""
    monkeypatch.setenv("EVEROS_DECIDER__EXTRA", _NO_THINK)
    assert run._decider_extra(_cfg()) == {
        "chat_template_kwargs": {"enable_thinking": False}
    } or run._decider_extra(_cfg()) == {
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
    }


def test_malformed_extra_does_not_crash_the_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The servers ignore unparseable JSON, so the probe must match that, loudly."""
    assert (
        run._decider_extra(_cfg(retrieval_env={"EVEROS_DECIDER__EXTRA": "{oops"})) == {}
    )


def test_thinking_only_reply_is_named_as_such(monkeypatch: pytest.MonkeyPatch) -> None:
    """ "Thought instead of answering" has a different fix from "said nothing"."""

    class Thinking(_Client):
        def _reply(self) -> _Resp:
            return _Resp("", reasoning="Let me consider the question...")

    monkeypatch.setattr(run.openai, "OpenAI", Thinking)
    with pytest.raises(run.DeciderUnreachableError, match="thinking is still on"):
        run._assert_decider_answers(_cfg())
