"""The eight extraction stages, each its message templates, language rules and reply model.

INIT, UPDATE, COMPACT and REGROUP on each path: the MemCell path's prompts are one user message, the Episode path's
a system message of rules and a user message of data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from everos.algo.llm import LLMMessage, StrictOutputModel
from everos.algo.user_memory._language import (
    COMPACTED_PROFILE_LANGUAGE_RULE,
    EPISODE_PROFILE_INIT_LANGUAGE_RULE,
    EXISTING_PROFILE_LANGUAGE_RULE,
    MEMCELL_COMPACTED_PROFILE_LANGUAGE_RULE,
    MEMCELL_EXISTING_PROFILE_LANGUAGE_RULE,
    MEMCELL_REGROUPED_PROFILE_LANGUAGE_RULE,
    PROFILE_INIT_LANGUAGE_RULE,
    REGROUPED_PROFILE_LANGUAGE_RULE,
    OutputLanguage,
    build_language_rule,
)
from everos.algo.user_memory.profile._replies import (
    _CompactOutput,
    _MemCellCompactOutput,
    _MemCellRegroupOutput,
    _MemCellUpdateOutput,
    _ProfileItems,
    _RegroupOutput,
    _UpdateOutput,
)
from everos.algo.user_memory.prompts.en.profile import (
    PROFILE_COMPACT_PROMPT,
    PROFILE_INITIAL_EXTRACTION_PROMPT,
    PROFILE_REGROUP_PROMPT,
    PROFILE_UPDATE_PROMPT,
    additional_requirements_section,
)
from everos.algo.user_memory.prompts.en.profile_from_episode_texts import (
    PROFILE_COMPACT_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
    PROFILE_COMPACT_FROM_EPISODE_TEXTS_USER_PROMPT,
    PROFILE_INITIAL_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
    PROFILE_INITIAL_FROM_EPISODE_TEXTS_USER_PROMPT,
    PROFILE_REGROUP_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
    PROFILE_REGROUP_FROM_EPISODE_TEXTS_USER_PROMPT,
    PROFILE_UPDATE_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
    PROFILE_UPDATE_FROM_EPISODE_TEXTS_USER_PROMPT,
    additional_requirements_block,
)
from everos.algo.utils import render_prompt


@dataclass(frozen=True, kw_only=True)
class _StagePrompts[ReplyT: StrictOutputModel]:
    """Message templates of one extraction stage, its language rules, and the model its reply is validated against.

    Args:
        user: Template of the user message. With no ``system`` it is the whole prompt (the MemCell path), and it
            carries the ``{{additional_requirements}}`` slot.
        reply: Output model of the stage's reply — the shape the stage's reply skeleton shows the model.
        language_fallback: Output-language rule used when the caller names no language, naming the input the
            stage takes its language from.
        system: Template of the system message holding the rules; free of per-request data.
    """

    user: str
    reply: type[ReplyT]
    language_fallback: str
    system: str | None = None

    def language_rule(self, output_language: OutputLanguage | str | None) -> str:
        """The output-language rule for ``output_language``: the caller-chosen rule, else this stage's fallback.

        Raises:
            InvalidInputError: If ``output_language`` names no known language.
        """
        return build_language_rule(output_language, fallback=self.language_fallback)

    def render(
        self, *, custom_instructions: str | None, **fields: Any
    ) -> list[LLMMessage]:
        """Render the system message (where the stage has one) and the user message from the same ``fields``.

        ``custom_instructions`` land after the rules and before the data, as text rather than template; None or a
        blank string adds nothing.
        """
        requirements = (
            custom_instructions
            if custom_instructions is not None and custom_instructions.strip()
            else None
        )
        if self.system is None:
            section = (
                additional_requirements_section(requirements)
                if requirements is not None
                else ""
            )
            user = render_prompt(
                self.user, None, **fields, additional_requirements=section
            )
            return [LLMMessage(role="user", content=user)]
        system = render_prompt(self.system, None, **fields)
        if requirements is not None:
            system = f"{system}\n\n{additional_requirements_block(requirements)}"
        user = render_prompt(self.user, None, **fields)
        return [
            LLMMessage(role="system", content=system),
            LLMMessage(role="user", content=user),
        ]


_MEMCELL_INIT = _StagePrompts(
    user=PROFILE_INITIAL_EXTRACTION_PROMPT,
    reply=_ProfileItems,
    language_fallback=PROFILE_INIT_LANGUAGE_RULE,
)
_MEMCELL_UPDATE = _StagePrompts(
    user=PROFILE_UPDATE_PROMPT,
    reply=_MemCellUpdateOutput,
    language_fallback=MEMCELL_EXISTING_PROFILE_LANGUAGE_RULE,
)
_MEMCELL_COMPACT = _StagePrompts(
    user=PROFILE_COMPACT_PROMPT,
    reply=_MemCellCompactOutput,
    language_fallback=MEMCELL_COMPACTED_PROFILE_LANGUAGE_RULE,
)
_MEMCELL_REGROUP = _StagePrompts(
    user=PROFILE_REGROUP_PROMPT,
    reply=_MemCellRegroupOutput,
    language_fallback=MEMCELL_REGROUPED_PROFILE_LANGUAGE_RULE,
)
_EPISODE_INIT = _StagePrompts(
    user=PROFILE_INITIAL_FROM_EPISODE_TEXTS_USER_PROMPT,
    reply=_ProfileItems,
    language_fallback=EPISODE_PROFILE_INIT_LANGUAGE_RULE,
    system=PROFILE_INITIAL_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
)
_EPISODE_UPDATE = _StagePrompts(
    user=PROFILE_UPDATE_FROM_EPISODE_TEXTS_USER_PROMPT,
    reply=_UpdateOutput,
    language_fallback=EXISTING_PROFILE_LANGUAGE_RULE,
    system=PROFILE_UPDATE_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
)
_EPISODE_COMPACT = _StagePrompts(
    user=PROFILE_COMPACT_FROM_EPISODE_TEXTS_USER_PROMPT,
    reply=_CompactOutput,
    language_fallback=COMPACTED_PROFILE_LANGUAGE_RULE,
    system=PROFILE_COMPACT_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
)
_EPISODE_REGROUP = _StagePrompts(
    user=PROFILE_REGROUP_FROM_EPISODE_TEXTS_USER_PROMPT,
    reply=_RegroupOutput,
    language_fallback=REGROUPED_PROFILE_LANGUAGE_RULE,
    system=PROFILE_REGROUP_FROM_EPISODE_TEXTS_SYSTEM_PROMPT,
)
