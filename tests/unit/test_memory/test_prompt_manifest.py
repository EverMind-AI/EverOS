"""The prompt manifest: which prompts this EverOS sends (``everos prompts``).

What these pin:

- every :class:`PromptTemplate` defined anywhere under ``everos.memory`` is in
  the manifest, under its own id, with its version and fingerprint -- a prompt
  registered later cannot be left out of what a store records;
- the everalgo prompts a chat store is written with are recorded by
  fingerprint against the vendored everalgo builds: the batch boundary
  detection (everalgo-boundary) every conversation is cut with, the episode
  rules ``chat.episode`` opens with, and the profile update and consolidation
  of ``ProfileExtractor.aupdate`` (everalgo-user-memory). An everalgo change to
  any of them fails here, on purpose, before it reaches a store;
- each entry says when its prompt is sent: while a store is written, on every
  round of every search (the multi-round decider, ``read``), or by an optional
  search step; a prompt assembled by rule lists each variant's fingerprint
  (``chat.episode`` and ``search.decider``: ``text`` and ``pictures``);
- the decider's ``text`` variant is ``_DECIDER_PROMPT`` alone, so its
  fingerprint is the first 16 hex digits of the SHA-256 of that constant --
  the digest the RL tooling already records for the prompt a policy saw.
"""

from __future__ import annotations

import hashlib
import importlib
import pkgutil
from pathlib import Path

import everos.memory
from everos.memory.prompt_manifest import (
    ALGO_PROMPTS,
    CUSTOM_INSTRUCTIONS,
    prompt_manifest,
)
from everos.memory.prompts import PromptTemplate
from everos.memory.search import llm_multiround


def _registered_templates() -> dict[str, PromptTemplate]:
    found: dict[str, PromptTemplate] = {}
    for info in pkgutil.walk_packages(everos.memory.__path__, "everos.memory."):
        module = importlib.import_module(info.name)
        for value in vars(module).values():
            if isinstance(value, PromptTemplate):
                found[value.id] = value
    return found


def test_every_registered_prompt_is_in_the_manifest() -> None:
    registered = _registered_templates()
    prompts = prompt_manifest()["prompts"]

    assert set(registered) <= set(prompts)
    for template in registered.values():
        entry = prompts[template.id]
        assert (entry["owner"], entry["version"], entry["fingerprint"]) == (
            "everos",
            template.version,
            template.fingerprint,
        )


def test_the_everalgo_prompts_are_recorded_against_the_vendored_build() -> None:
    prompts = prompt_manifest()["prompts"]
    recorded = {
        prompt.id: (
            prompts[prompt.id]["package"],
            prompts[prompt.id]["package_version"],
            prompts[prompt.id]["fingerprint"],
        )
        for prompt in ALGO_PROMPTS
    }

    assert recorded == {
        "everalgo.boundary.batch_detect": (
            "everalgo-boundary",
            "0.4.1",
            "db589ab87364c7db",
        ),
        "everalgo.user_memory.episode_rules": (
            "everalgo-user-memory",
            "0.9.0rc6.dev6",
            "0c173343b7149d58",
        ),
        "everalgo.user_memory.profile_update": (
            "everalgo-user-memory",
            "0.9.0rc6.dev6",
            "6cc9e75ee44c2c98",
        ),
        "everalgo.user_memory.profile_consolidate": (
            "everalgo-user-memory",
            "0.9.0rc6.dev6",
            "d70ed25f5e44ede3",
        ),
        "everalgo.user_memory.language_rules": (
            "everalgo-user-memory",
            "0.9.0rc6.dev6",
            "10d1df0535db08bf",
        ),
        "everalgo.parser.prompts": (
            "everalgo-parser",
            "0.3.0rc2",
            "e076202ae1deaeed",
        ),
    }


def test_the_episode_prompt_is_recorded_with_its_two_variants() -> None:
    episode = prompt_manifest()["prompts"]["chat.episode"]

    assert (episode["version"], episode["used"]) == (2, "write")
    assert episode["variants"] == {
        "text": "2117ea4fa75e421a",
        "pictures": "bae4c0c544622565",
    }


def test_each_entry_says_when_its_prompt_is_sent() -> None:
    prompts = prompt_manifest()["prompts"]

    assert prompts["chat.scene_update"]["used"] == "write"
    assert prompts["everalgo.boundary.batch_detect"]["used"] == "write"
    assert prompts["search.constraint_bind"]["used"] == "search"
    assert prompts["search.decider"]["used"] == "read"


def test_the_decider_prompt_is_recorded_with_its_two_variants() -> None:
    decider = prompt_manifest()["prompts"]["search.decider"]

    assert (decider["owner"], decider["version"]) == ("everos", 1)
    assert decider["fingerprint"] == "30088db20a1bbf50"
    assert decider["variants"] == {
        "text": "4662b538083707a2",
        "pictures": "30088db20a1bbf50",
    }


def test_the_decider_text_variant_is_the_prompt_constant_alone() -> None:
    prompt = llm_multiround._DECIDER_PROMPT
    text = prompt_manifest()["prompts"]["search.decider"]["variants"]["text"]

    assert llm_multiround.DECIDER.parts[0] is prompt
    assert text == hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def test_the_custom_instructions_are_recorded_off_or_by_their_text(
    tmp_path: Path,
) -> None:
    """Off (the bundled files), an entry records that nothing is sent; on, the
    fingerprint of the text appended to its prompt."""
    bundled = prompt_manifest()["prompts"]
    for operator, appended_to in CUSTOM_INSTRUCTIONS:
        entry = bundled[f"custom_instructions.{operator}"]
        assert entry == {
            "owner": "config",
            "used": "write",
            "appended_to": appended_to,
            "enabled": False,
            "fingerprint": None,
        }
        assert appended_to in bundled

    folder = tmp_path / "custom_instructions"
    folder.mkdir()
    (folder / "episode_extract.yaml").write_text(
        'enabled: true\ninstructions: "Keep every date."\n', encoding="utf-8"
    )
    first = prompt_manifest(config_root=tmp_path)["prompts"]
    (folder / "episode_extract.yaml").write_text(
        'enabled: true\ninstructions: "Keep every price."\n', encoding="utf-8"
    )
    second = prompt_manifest(config_root=tmp_path)["prompts"]
    on = first["custom_instructions.episode_extract"]
    assert on["enabled"] is True and on["fingerprint"]
    assert (
        second["custom_instructions.episode_extract"]["fingerprint"]
        != on["fingerprint"]
    )
    assert first["custom_instructions.reflect"]["enabled"] is False
