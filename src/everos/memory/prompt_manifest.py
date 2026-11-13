"""Which prompts this EverOS sends: the manifest a store's builder records.

Two kinds of prompt build and read a chat store, and both are listed with the
fingerprint of their text:

* EverOS's own (:class:`~everos.memory.prompts.PromptTemplate`, id, version
  and fingerprint pinned by tests): the episode prompt every chat cell is
  extracted with, the group roster block, the group-profile update, the scene
  update, the constraint binding a group search may call, and the multi-round
  decider every search round sends.
* everalgo's, sent as the installed package wrote them
  (:class:`~everos.memory.prompts.AlgoPrompt`): the boundary detection that
  cuts a conversation into segments, the episode rules the episode prompt
  opens with, the profile update and consolidation, the output-language rules
  every extraction prompt carries in its ``<output_language>`` block, and the
  parser prompts that read a picture, a document or a recording into text
  (``[extraction] vision = false``, ``[multimodal]``).
* the custom instructions (``config/custom_instructions/*.yaml``) an operator
  appends to the boundary, episode, profile and scene prompts: each with
  whether it is on and, when it is, the fingerprint of the text sent.

Each entry says when its prompt is sent: ``write`` while a store is written,
``read`` by the multi-round decider on every round of every search (the read
policy, the prompt a trained decider is trained on), ``search`` by an optional
search step.

``prompt_manifest()`` returns them as one JSON-ready mapping, the shape
``everos prompts`` prints. A store records which prompts built it by saving
that output when it is built; two stores built from the same prompts carry
the same mapping. Recording it fixes nothing: the prompts stay free to change,
and a change shows up as a new fingerprint (and, for EverOS's own, a new
version).
"""

from __future__ import annotations

__all__ = [
    "ALGO_PROMPTS",
    "CUSTOM_INSTRUCTIONS",
    "EVEROS_PROMPTS",
    "MANIFEST_VERSION",
    "prompt_manifest",
]

from importlib import metadata
from pathlib import Path
from typing import Any

import everalgo.parser.prompts.en as parser_prompts
import everalgo.user_memory._language as language_rules
from everalgo.boundary.prompts.en.chat import (
    BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT,
    BATCH_BOUNDARY_DETECT_USER_PROMPT,
)
from everalgo.user_memory.prompts.en.profile import (
    PROFILE_CONSOLIDATE_SYSTEM_PROMPT,
    PROFILE_CONSOLIDATE_USER_PROMPT,
    PROFILE_UPDATE_SYSTEM_PROMPT,
    PROFILE_UPDATE_USER_PROMPT,
)

import everos.config
from everos.memory.custom_instructions import CustomInstructions, Operator
from everos.memory.extract.group_profile import GROUP_PROFILE_UPDATE
from everos.memory.extract.pipeline.episode import CHAT_EPISODE, EPISODE_RULES
from everos.memory.extract.pipeline.group_render import ROSTER_BLOCK
from everos.memory.extract.scene import SCENE_UPDATE
from everos.memory.prompts import AlgoPrompt, PromptTemplate
from everos.memory.search.constraints import CONSTRAINT_BIND
from everos.memory.search.llm_multiround import DECIDER

MANIFEST_VERSION = 2
"""Version of the manifest's own shape: 2 added the custom instructions'
entries (``owner: config``) beside the prompts."""

EVEROS_PROMPTS: tuple[tuple[PromptTemplate, str], ...] = (
    (CHAT_EPISODE, "write"),
    (ROSTER_BLOCK, "write"),
    (GROUP_PROFILE_UPDATE, "write"),
    (SCENE_UPDATE, "write"),
    (CONSTRAINT_BIND, "search"),
    (DECIDER, "read"),
)
"""EverOS's registered prompts, each with when it is sent: while a store is
written (``write``), on every round of every search (``read``), or by an
optional search step (``search``)."""

ALGO_PROMPTS: tuple[AlgoPrompt, ...] = (
    AlgoPrompt(
        id="everalgo.boundary.batch_detect",
        package="everalgo-boundary",
        parts=(BATCH_BOUNDARY_DETECT_SYSTEM_PROMPT, BATCH_BOUNDARY_DETECT_USER_PROMPT),
    ),
    AlgoPrompt(
        id="everalgo.user_memory.episode_rules",
        package="everalgo-user-memory",
        parts=(EPISODE_RULES,),
    ),
    AlgoPrompt(
        id="everalgo.user_memory.profile_update",
        package="everalgo-user-memory",
        parts=(PROFILE_UPDATE_SYSTEM_PROMPT, PROFILE_UPDATE_USER_PROMPT),
    ),
    AlgoPrompt(
        id="everalgo.user_memory.profile_consolidate",
        package="everalgo-user-memory",
        parts=(PROFILE_CONSOLIDATE_SYSTEM_PROMPT, PROFILE_CONSOLIDATE_USER_PROMPT),
    ),
    AlgoPrompt(
        id="everalgo.user_memory.language_rules",
        package="everalgo-user-memory",
        parts=tuple(
            getattr(language_rules, name)
            for name in sorted(language_rules.__all__)
            if name.endswith("_LANGUAGE_RULE")
        ),
    ),
    AlgoPrompt(
        id="everalgo.parser.prompts",
        package="everalgo-parser",
        parts=tuple(
            getattr(parser_prompts, name) for name in sorted(parser_prompts.__all__)
        ),
    ),
)
"""The everalgo prompts a chat store is written with, all sent while it is
written: the boundary detection is ``adetect_boundaries``' batch call, which
every buffered conversation goes through before it is extracted; the episode
rules open ``chat.episode``; the two profile prompts are everalgo
``ProfileExtractor.aupdate``'s; the language rules fill the
``<output_language>`` block of every extraction prompt (the episode's, the
profile's, a group profile's, a scene's, the constraint binding's); the parser
prompts read a picture (when the extraction model reads none), a document or
a recording into text."""

CUSTOM_INSTRUCTIONS: tuple[tuple[Operator, str], ...] = (
    ("boundary_detection", "everalgo.boundary.batch_detect"),
    ("episode_extract", "chat.episode"),
    ("profile_extract", "everalgo.user_memory.profile_update"),
    ("reflect", "chat.scene_update"),
)
"""Each custom-instruction file and the prompt it is appended to."""


def prompt_manifest(*, config_root: Path | None = None) -> dict[str, Any]:
    """Every prompt above by id: owner, when it is sent, version, fingerprint.

    A prompt assembled by rule also lists each variant's fingerprint
    (``chat.episode`` and ``search.decider``: ``text`` and ``pictures``). An
    everalgo prompt carries its package and the version installed in place of
    an EverOS version. A custom-instruction entry names the prompt it is
    appended to and whether it is on; its fingerprint is ``None`` while off
    (nothing is sent). ``config_root`` holds the ``custom_instructions``
    folder; the bundled one (``src/everos/config``, what the service loads)
    when omitted.
    """
    prompts: dict[str, dict[str, Any]] = {}
    for template, used in EVEROS_PROMPTS:
        prompts[template.id] = {
            "owner": "everos",
            "used": used,
            "version": template.version,
            "fingerprint": template.fingerprint,
        }
        if template.variants:
            prompts[template.id]["variants"] = {
                name: template.variant_fingerprint(name)
                for name, _ in template.variants
            }
    for prompt in ALGO_PROMPTS:
        prompts[prompt.id] = {
            "owner": "everalgo",
            "used": "write",
            "package": prompt.package,
            "package_version": metadata.version(prompt.package),
            "fingerprint": prompt.fingerprint,
        }
    instructions = CustomInstructions(
        config_root or Path(everos.config.__file__).resolve().parent
    )
    for operator, appended_to in CUSTOM_INSTRUCTIONS:
        text = instructions.load(operator)
        prompts[f"custom_instructions.{operator}"] = {
            "owner": "config",
            "used": "write",
            "appended_to": appended_to,
            "enabled": text is not None,
            "fingerprint": None
            if text is None
            else AlgoPrompt(id=operator, package="", parts=(text,)).fingerprint,
        }
    return {"manifest_version": MANIFEST_VERSION, "prompts": prompts}
