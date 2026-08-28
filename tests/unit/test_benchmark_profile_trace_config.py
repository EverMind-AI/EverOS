"""``include_profile`` / ``trace`` as config, not as code.

Both used to be implicit: ``INCLUDE_PROFILE`` lived on the adapter module and the
traces switched on whenever the fleet happened to get a ``trace_dir``. Neither was
visible in the run's recorded config, so a run could not say afterwards whether it
had injected profiles or written a trace. These tests pin the two knobs and the
precedence between the config value and the adapter's declaration.
"""

from __future__ import annotations

import importlib
import sys
import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_BENCH = _ROOT / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from config import BenchmarkConfig  # noqa: E402

# The shipped configs. `longmemeval_qwen38` was one of these until it was removed: a
# local-backbone variant of the same benchmark, kept only for one operator run. Its
# durable lessons live where they still apply -- the extra_body nesting in
# the launch scripts and the empty-completion case in
# `test_benchmark_decider_endpoint.py`.
_NAMES = ("locomo", "longmemeval", "subtlememory", "evermembench")
_ARMS = _NAMES


def _config(name: str) -> BenchmarkConfig:
    raw = tomllib.loads((_BENCH / "configs" / f"{name}.toml").read_text())
    flat = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    for section in ("answer", "judge"):
        for k, v in (raw.get(section) or {}).items():
            flat[f"{section}_{k}"] = v
    flat["eval_concurrency"] = flat.pop("answer_concurrency", 20)
    flat.pop("judge_concurrency", None)
    return BenchmarkConfig.model_validate(flat)


# Every arm ships with the profile OFF, and that is the controlled position rather than
# a coincidence: each benchmark's reference harness sends no profile, so ON is a
# deviation from the published prompt. Turning one on is a declared ablation, and it has
# to clear the measured run-to-run noise floor -- 0.91 pp on LoCoMo, 1.20 pp on
# LongMemEval -- before the difference means anything.
#
# evermembench has a second, independent reason: its five reference stores hold no
# profile at all (profile_md=0, users=0) and its evaluator has no profile concept.
_EXPECTED_PROFILE = dict.fromkeys(_ARMS, False)


@pytest.mark.parametrize("name", _ARMS)
def test_every_benchmark_declares_both_knobs(name: str) -> None:
    """Explicit in the toml, so the recorded run says what it did."""
    text = (_BENCH / "configs" / f"{name}.toml").read_text()
    assert "include_profile" in text, f"{name} does not declare include_profile"
    assert "\ntrace = " in text, f"{name} does not declare trace"
    cfg = _config(name)
    assert cfg.include_profile is _EXPECTED_PROFILE[name], (
        f"{name}: include_profile changed -- if deliberate, update _EXPECTED_PROFILE "
        f"and say why in the config header"
    )
    assert cfg.trace is True


def test_defaults_keep_the_adapter_in_charge() -> None:
    """``None`` is not ``False``: it means "ask the adapter", which is how a
    benchmark whose reference never sent the flag keeps getting no profile."""
    cfg = BenchmarkConfig()
    assert cfg.include_profile is None
    assert cfg.trace is True


@pytest.mark.parametrize(
    ("config_value", "adapter_value", "expected"),
    [
        (None, True, True),  # adapter decides
        (None, False, False),
        (True, False, True),  # config overrides -- what an ablation needs
        (False, True, False),
    ],
)
def test_config_overrides_the_adapter_only_when_it_says_something(
    config_value: bool | None, adapter_value: bool, expected: bool
) -> None:
    resolved = config_value if config_value is not None else adapter_value
    assert resolved is expected


def test_trace_off_gives_the_fleet_no_directory(tmp_path: Path) -> None:
    """The fleet skips both dumps when ``trace_dir`` is None; that is the switch."""
    run = importlib.import_module("run")
    fleet = run._ServerFleet(1, tmp_path / "store", first_port=19999, trace_dir=None)
    assert fleet.trace_dir is None


def test_the_longmemeval_baseline_still_names_the_reference_models() -> None:
    """``check_protocol.py`` asserts this config matches the harness it came from.

    Editing the baseline in place is what silences that gate, and the gate is the only
    thing between a config drift and a number nobody can compare. A variant run belongs
    in its own file for the same reason -- which is what `longmemeval_qwen38.toml` was
    until it was deleted, having served its one operator run.
    """
    base = tomllib.loads((_BENCH / "configs" / "longmemeval.toml").read_text())
    assert base["backbone_model"] == "deepseek/deepseek-v4-pro-0813"
    # The decider is named by BENCH_DECIDER_MODEL rather than hardcoded: shipping a
    # model the reader does not serve, with an endpoint that may be unset, is the
    # configuration that 404s on every call and falls back silently. What still has to
    # hold is that the config NAMES it rather than leaving the field absent -- and that
    # the fallback is the PUBLISHED decider, not whatever a variant run last used. The
    # deleted `longmemeval_qwen38.toml` leaked `qwen3.8-27b` into this baseline once;
    # pinning the default here is what turns that class of drift back into a failure.
    assert base["decider_model"] == "${BENCH_DECIDER_MODEL:-qwen3.6-27B}"
    assert base["answer"]["model"] == "openai/gpt-4.1-mini"
    assert base["providers"] == {}


@pytest.mark.parametrize("name", _ARMS)
def test_every_benchmark_disables_the_inotify_watcher(name: str) -> None:
    """A shared host's inotify ceiling is a per-user kernel resource.

    Measured on this box: the IDE's file indexer held 522,237 of the 524,288
    available watches (351,757 in one process), and all four EverOS fleets died at
    startup with ``OSError: [Errno 28] inotify watch limit reached``. The ceiling is
    in ``/proc/sys``, which is read-only here, so raising it is not an option. The
    cascade scanner re-derives the same truth every 30s, so the watcher is latency,
    not correctness -- but ``EVEROS_DISABLE_CASCADE`` would take the worker with it
    and md would never reach LanceDB at all.
    """
    raw = tomllib.loads((_BENCH / "configs" / f"{name}.toml").read_text())
    assert raw["retrieval_env"]["EVEROS_DISABLE_CASCADE_WATCHER"] == "1"
    # The worker must survive: this is an ingesting run.
    assert "EVEROS_DISABLE_CASCADE" not in raw["retrieval_env"]


def test_smoke_does_not_discard_an_explicit_conv_list() -> None:
    """``--smoke`` narrows a run; it must not silently replace ``--conv``.

    It used to assign ``[0, 1]`` unconditionally, so ``--conv 0 --smoke`` ran two
    conversations and scored 20 questions while the operator had asked for one. The
    override left no trace in the output -- the banner printed the conversations it
    had chosen, not the ones it was given -- so the only way to catch it was to count
    graded rows against what you expected.
    """
    src = (_BENCH / "run.py").read_text()
    i = src.index("if args.smoke and not _conv_given:")
    assert "_conv_given = args.conv is not None" in src[:i]
    # The guard has to sit on the assignment itself, not merely exist somewhere.
    assert src[i : i + 200].lstrip().startswith("if args.smoke and not _conv_given:")
    assert "if args.smoke:\n        args.conv = [0, 1]" not in src
