"""``configs/locomo-step.toml`` is ``locomo.toml`` with two server settings changed.

The step variant is published next to the batch number, so the two must measure the
same thing apart from what the variant names: step boundary detection and
prompt-only replies. A copied config drifts silently when only one side is edited; this
pins the difference to exactly those two ``[retrieval_env]`` entries.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks"))

from config import INGEST_SERVER_SETTINGS

_CONFIGS = Path(__file__).resolve().parents[2] / "benchmarks" / "configs"
_STEP_ONLY = {
    "EVEROS_BOUNDARY_DETECTION__STRATEGY": "step",
    "EVEROS_LLM__RESPONSE_FORMAT_MODE": "prompt_only",
}


def _load(name: str) -> dict:
    return tomllib.loads((_CONFIGS / f"{name}.toml").read_text())


def test_the_step_config_differs_from_locomo_only_in_its_two_server_settings() -> None:
    batch, step = _load("locomo"), _load("locomo-step")

    step_env = dict(step.pop("retrieval_env"))
    batch_env = dict(batch.pop("retrieval_env"))
    added = {k: step_env.pop(k) for k in _STEP_ONLY}

    assert added == _STEP_ONLY
    assert step_env == batch_env
    assert step == batch


def test_both_step_settings_are_part_of_the_ingest_identity() -> None:
    """A store built by one variant must not be reused by the other."""
    assert set(_STEP_ONLY) <= set(INGEST_SERVER_SETTINGS.values())
