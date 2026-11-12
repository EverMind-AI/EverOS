"""The prompt manifest: which prompts this EverOS sends (``everos prompts``).

What these pin:

- every :class:`PromptTemplate` defined anywhere under ``everos.memory`` is in
  the manifest, under its own id, with its version and fingerprint -- a prompt
  registered later cannot be left out of what a store records;
- the everalgo prompts a chat store is written with are recorded by
  fingerprint against the vendored everalgo-user-memory build: the episode
  rules ``chat.episode`` opens with, and the profile update and consolidation
  of ``ProfileExtractor.aupdate``. An everalgo change to any of them fails
  here, on purpose, before it reaches a store;
- each entry says when its prompt is sent: while a store is written, or while
  it is searched; a prompt assembled by rule lists each variant's fingerprint
  (``chat.episode``: ``text`` and ``pictures``).
"""

from __future__ import annotations

import importlib
import pkgutil

import everos.memory
from everos.memory.prompt_manifest import ALGO_PROMPTS, prompt_manifest
from everos.memory.prompts import PromptTemplate


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
    assert prompts["search.constraint_bind"]["used"] == "search"
