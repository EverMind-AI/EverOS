"""``ProfileExtractor``: the two entry paths and the stages each runs, wired from the sibling modules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from everos.algo.user_memory._language import OutputLanguage, resolve_output_language
from everos.algo.user_memory.profile._items import (
    _ITEM_WIDTH_BACKSTOP,
    _LABEL_FIELDS,
    _PROFILE_MAX_ITEMS,
    _PROFILE_MAX_PER_CATEGORY,
    _in_group,
    _Item,
    _item_label,
    _normalize,
)
from everos.algo.user_memory.profile._merge import (
    _apply_ops,
    _dedupe,
    _identity_key,
    _overcrowded_labels,
    _profile_from_buckets,
    _stored_items,
)
from everos.algo.user_memory.profile._render import (
    _normalize_available_categories,
    _render_available_categories,
    _render_conversation,
    _render_episodes,
    _render_group,
    _render_profile_for_compact,
    _render_profile_for_episode_update,
    _render_profile_for_update,
    _resolve_episode_target,
    _select_episodes,
    _sender_display_name,
    _user_senders,
)
from everos.algo.user_memory.profile._replies import (
    _AnyCompactOutput,
    _AnyRegroupOutput,
    _call_llm_for_profile_compact,
    _call_llm_for_profile_init,
    _call_llm_for_profile_regroup,
    _call_llm_for_profile_update,
)
from everos.algo.user_memory.profile._stages import (
    _EPISODE_COMPACT,
    _EPISODE_INIT,
    _EPISODE_REGROUP,
    _EPISODE_UPDATE,
    _MEMCELL_COMPACT,
    _MEMCELL_INIT,
    _MEMCELL_REGROUP,
    _MEMCELL_UPDATE,
    _StagePrompts,
)
from everos.algo.user_memory.prompts.en.profile_from_episode_texts import (
    REGROUP_REPLY_SKELETONS,
)
from everos.algo.utils import InvalidInputError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from everos.algo.llm.protocols import LLMClient
    from everos.algo.types import Episode, MemCell, Profile

# The package's name rather than this module's: hosts collect the log contract under it.
logger = logging.getLogger("everos.algo.user_memory.profile")


class ProfileExtractor:
    """Synthesize one owner-scoped Profile from chronological memory inputs.

    ``aextract`` reads chronologically ordered MemCells and silently skips non-ChatMessage items.
    ``aextract_from_episodes`` reads dated generic or reflected Episodes in any order, skips items that do not
    reference the resolved owner, raises only when none reference that owner, and keeps the profile's timeline
    itself: an Episode observed before the stored profile's newest date can add to the profile but never
    rewrites or deletes what a later Episode established.

    Args:
        llm: The client every stage calls.
    """

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    async def aextract(
        self,
        memcells: Sequence[MemCell],
        *,
        sender_id: str,
        old_profile: Profile | None = None,
        custom_instructions: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Profile:
        """Extract one Profile for ``sender_id`` from ``memcells``.

        With no ``old_profile`` this is INIT, a full extraction. Otherwise it is UPDATE: the model emits
        add / update / delete ops on ``old_profile``, and a merged profile past an item cap gets one COMPACT or
        REGROUP pass, transparent to the caller. Exceptions raised by the LLM client propagate unchanged.

        Args:
            memcells: Non-empty and chronological; the last element is the most recent.
            sender_id: One of the memcells' human (``role == "user"``) senders, never an assistant. The prompt
                scopes extraction to this speaker.
            old_profile: Existing profile for UPDATE; None for INIT.
            custom_instructions: The host's instructions for every call this makes, added verbatim in an
                "Additional Requirements" section after the bundled rules and before the data. They cannot replace
                the rules or the output shape. None or a blank string adds nothing.
            output_language: Language to write the profile in, as an :class:`OutputLanguage` member or its value
                in any casing. None lets INIT follow the participants and UPDATE and COMPACT keep the language the
                profile is written in.

        Returns:
            The extracted or updated Profile.

        Raises:
            InvalidInputError: If ``memcells`` is empty, ``sender_id`` is not a user speaker in ``memcells``,
                ``output_language`` names no supported language, or a ChatMessage with text has no timestamp (0).
            LLMOutputError: If a reply is not a valid INIT, UPDATE, COMPACT or REGROUP output.
        """
        if not memcells:
            raise InvalidInputError(
                "memcells is empty, there is nothing to extract a profile from: "
                "pass at least one MemCell, or skip the call when a conversation has none"
            )

        user_senders = _user_senders(memcells)
        if sender_id not in user_senders:
            raise InvalidInputError(
                f"sender_id {sender_id!r} is not a user speaker in memcells (user senders: {sorted(user_senders)!r}): "
                "pass the sender_id of a human speaker (role 'user') who appears in the memcells"
            )

        language = resolve_output_language(output_language)
        mode = "INIT" if old_profile is None else "UPDATE"
        logger.info(
            "extracting profile: mode=%s, %d memcells, existing explicit=%d implicit=%d, output_language=%s, "
            "custom_instructions=%d chars",
            mode,
            len(memcells),
            len(old_profile.explicit_info) if old_profile else 0,
            len(old_profile.implicit_traits) if old_profile else 0,
            language,
            len(custom_instructions or ""),
        )
        if old_profile is None:
            result = await self._init_extract(
                memcells,
                sender_id=sender_id,
                custom_instructions=custom_instructions,
                output_language=language,
            )
        else:
            result = await self._update_extract(
                memcells,
                sender_id=sender_id,
                old_profile=old_profile,
                custom_instructions=custom_instructions,
                output_language=language,
            )
        logger.info(
            "profile extracted: mode=%s -> explicit=%d implicit=%d",
            mode,
            len(result.explicit_info),
            len(result.implicit_traits),
        )
        return result

    async def aextract_from_episodes(
        self,
        episodes: Sequence[Episode],
        *,
        owner_id: str,
        owner_name: str | None = None,
        old_profile: Profile | None = None,
        categories: Sequence[str] | None = None,
        custom_instructions: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Profile:
        """Extract one owner-scoped Profile from dated Episode narratives.

        Each Episode's ``timestamp`` is the observation date of what it narrates. The batch is ordered by that
        date, and in UPDATE mode split around the stored ``Profile.timestamp``: Episodes observed before it may
        add facts and evidence but never rewrite or delete what a later Episode established; the rest update
        normally. Callers pass whatever Episodes are new to them, in any order. Exceptions raised by the LLM client
        propagate unchanged.

        Args:
            episodes: Non-empty Episodes with non-blank narratives. Items that do not contain the resolved target
                reference are excluded; at least one must contain it.
            owner_id: Authoritative owner identifier written to ``Profile.owner_id``.
            owner_name: Display name used to select narratives about the owner; blank or missing falls back to
                ``owner_id``.
            old_profile: Existing profile for UPDATE; None for INIT.
            categories: Complete current category snapshot for ``explicit_info``, stripped, blanks ignored and
                exact duplicates removed in first-seen order. None or empty lets the model coin categories. It
                does not constrain ``implicit_traits.trait``.
            custom_instructions: The host's instructions for every call this makes, added verbatim as an
                ``<additional_requirements>`` block closing the system message. They cannot replace the rules or the
                output shape. None or a blank string adds nothing.
            output_language: Language to write the profile in. None makes INIT inherit the Episode language and
                UPDATE preserve the existing Profile language.

        Returns:
            The extracted or updated Profile. ``Profile.timestamp`` is the newest observation date seen so far and
            never moves backwards; every item carries ``observed_at``, the observation date of the narrative that
            last established its description.

        Raises:
            TypeError: If an item of ``episodes`` is not an :class:`Episode`, or ``categories`` is not a sequence
                of strings or contains a non-string value.
            InvalidInputError: If owner or Episode validation fails, ``old_profile`` belongs to another owner, or
                ``output_language`` names no supported language.
            LLMOutputError: If a reply is not a valid INIT, UPDATE, COMPACT or REGROUP output.
        """
        target_user = _resolve_episode_target(owner_id, owner_name)
        selected = _select_episodes(episodes, target_user=target_user)
        category_snapshot = _normalize_available_categories(categories)
        available_categories = _render_available_categories(category_snapshot)
        if old_profile is not None and old_profile.owner_id != owner_id:
            raise InvalidInputError(
                f"old_profile.owner_id {old_profile.owner_id!r} does not match owner_id {owner_id!r}: "
                "pass the stored profile of the same owner, or None to start a new one"
            )
        language = resolve_output_language(output_language)

        mode = "INIT" if old_profile is None else "UPDATE"
        logger.info(
            "extracting profile from Episodes: mode=%s, %d/%d matching episodes observed at ms %d..%d, "
            "existing explicit=%d implicit=%d, output_language=%s, categories=%d, custom_instructions=%d chars",
            mode,
            len(selected),
            len(episodes),
            selected[0].timestamp,
            selected[-1].timestamp,
            len(old_profile.explicit_info) if old_profile else 0,
            len(old_profile.implicit_traits) if old_profile else 0,
            language,
            len(category_snapshot),
            len(custom_instructions or ""),
        )
        if old_profile is None:
            result = await self._init_extract_from_episodes(
                selected,
                owner_id=owner_id,
                target_user=target_user,
                available_categories=available_categories,
                custom_instructions=custom_instructions,
                output_language=language,
            )
        else:
            result = await self._update_extract_from_episodes(
                selected,
                owner_id=owner_id,
                target_user=target_user,
                old_profile=old_profile,
                available_categories=available_categories,
                custom_instructions=custom_instructions,
                output_language=language,
            )
        logger.info(
            "profile extracted from Episodes: mode=%s -> explicit=%d implicit=%d",
            mode,
            len(result.explicit_info),
            len(result.implicit_traits),
        )
        return result

    # ------------------------------------------------------------------
    # Private: INIT path
    # ------------------------------------------------------------------

    async def _init_extract(
        self,
        memcells: Sequence[MemCell],
        *,
        sender_id: str,
        custom_instructions: str | None,
        output_language: OutputLanguage | str | None,
    ) -> Profile:
        messages = _MEMCELL_INIT.render(
            custom_instructions=custom_instructions,
            conversation_text=_render_conversation(memcells),
            target_user=_sender_display_name(memcells, sender_id),
            target_user_id=sender_id,
            language_rule=_MEMCELL_INIT.language_rule(output_language),
        )

        data = await _call_llm_for_profile_init(
            self._llm, messages, reply=_MEMCELL_INIT.reply
        )
        observed_at = memcells[-1].timestamp
        return _profile_from_buckets(
            owner_id=sender_id,
            timestamp=observed_at,
            explicit_info=_dedupe(
                _stored_items("explicit_info", data["explicit_info"], observed_at),
                source="init",
            ),
            implicit_traits=_dedupe(
                _stored_items("implicit_traits", data["implicit_traits"], observed_at),
                source="init",
            ),
        )

    async def _init_extract_from_episodes(
        self,
        episodes: Sequence[Episode],
        *,
        owner_id: str,
        target_user: str,
        available_categories: str,
        custom_instructions: str | None,
        output_language: OutputLanguage | str | None,
    ) -> Profile:
        messages = _EPISODE_INIT.render(
            custom_instructions=custom_instructions,
            episode_texts=_render_episodes(episodes),
            target_user=target_user,
            available_categories=available_categories,
            language_rule=_EPISODE_INIT.language_rule(output_language),
        )

        data = await _call_llm_for_profile_init(
            self._llm, messages, reply=_EPISODE_INIT.reply
        )
        observed_at = episodes[-1].timestamp
        return _profile_from_buckets(
            owner_id=owner_id,
            timestamp=observed_at,
            explicit_info=_dedupe(
                _stored_items("explicit_info", data["explicit_info"], observed_at),
                source="episode-init",
            ),
            implicit_traits=_dedupe(
                _stored_items("implicit_traits", data["implicit_traits"], observed_at),
                source="episode-init",
            ),
        )

    # ------------------------------------------------------------------
    # Private: UPDATE path
    # ------------------------------------------------------------------

    async def _update_extract(
        self,
        memcells: Sequence[MemCell],
        *,
        sender_id: str,
        old_profile: Profile,
        custom_instructions: str | None,
        output_language: OutputLanguage | str | None,
    ) -> Profile:
        messages = _MEMCELL_UPDATE.render(
            custom_instructions=custom_instructions,
            stored_profile=_render_profile_for_update(old_profile),
            conversations=_render_conversation(memcells),
            target_user=_sender_display_name(memcells, sender_id),
            target_user_id=sender_id,
            language_rule=_MEMCELL_UPDATE.language_rule(output_language),
        )

        data = await _call_llm_for_profile_update(
            self._llm, messages, reply=_MEMCELL_UPDATE.reply
        )
        merged_profile = _apply_ops(
            old_profile,
            data["operations"],
            timestamp=memcells[-1].timestamp,
            pass_name="single",
        )

        return await self._maintain_updated_profile(
            merged_profile,
            display_name=_sender_display_name(memcells, sender_id),
            sender_id=sender_id,
            output_language=output_language,
            custom_instructions=custom_instructions,
            compact_prompts=_MEMCELL_COMPACT,
            regroup_prompts=_MEMCELL_REGROUP,
            available_categories=None,
        )

    async def _update_extract_from_episodes(
        self,
        episodes: Sequence[Episode],
        *,
        owner_id: str,
        target_user: str,
        old_profile: Profile,
        available_categories: str,
        custom_instructions: str | None,
        output_language: OutputLanguage | str | None,
    ) -> Profile:
        """Run UPDATE in up to two passes: Episodes older than the stored profile first, then the rest.

        Each pass carries one observation date, its newest narrative, which ``_collect_indexed`` judges every
        rewrite against.
        """
        historical = [
            episode for episode in episodes if episode.timestamp < old_profile.timestamp
        ]
        current = list(episodes[len(historical) :])
        if historical:
            logger.info(
                "profile update split around stored timestamp ms %d: %d historical, %d current narratives",
                old_profile.timestamp,
                len(historical),
                len(current),
            )
        profile = old_profile
        for pass_name, batch in (("historical", historical), ("current", current)):
            if batch:
                profile = await self._update_pass_from_episodes(
                    batch,
                    pass_name=pass_name,
                    owner_id=owner_id,
                    target_user=target_user,
                    old_profile=profile,
                    available_categories=available_categories,
                    custom_instructions=custom_instructions,
                    output_language=output_language,
                )
        return profile

    async def _update_pass_from_episodes(
        self,
        episodes: Sequence[Episode],
        *,
        pass_name: str,
        owner_id: str,
        target_user: str,
        old_profile: Profile,
        available_categories: str,
        custom_instructions: str | None,
        output_language: OutputLanguage | str | None,
    ) -> Profile:
        messages = _EPISODE_UPDATE.render(
            custom_instructions=custom_instructions,
            stored_profile=_render_profile_for_episode_update(old_profile),
            episode_texts=_render_episodes(episodes),
            target_user=target_user,
            available_categories=available_categories,
            language_rule=_EPISODE_UPDATE.language_rule(output_language),
        )

        data = await _call_llm_for_profile_update(
            self._llm, messages, reply=_EPISODE_UPDATE.reply
        )
        merged_profile = _apply_ops(
            old_profile,
            data["operations"],
            timestamp=episodes[-1].timestamp,
            pass_name=pass_name,
        )
        return await self._maintain_updated_profile(
            merged_profile,
            display_name=target_user,
            sender_id=owner_id,
            output_language=output_language,
            custom_instructions=custom_instructions,
            compact_prompts=_EPISODE_COMPACT,
            regroup_prompts=_EPISODE_REGROUP,
            available_categories=available_categories,
        )

    async def _maintain_updated_profile[
        CompactT: _AnyCompactOutput,
        RegroupT: _AnyRegroupOutput,
    ](
        self,
        merged_profile: Profile,
        *,
        display_name: str,
        sender_id: str,
        output_language: OutputLanguage | str | None,
        custom_instructions: str | None,
        compact_prompts: _StagePrompts[CompactT],
        regroup_prompts: _StagePrompts[RegroupT],
        available_categories: str | None = None,
    ) -> Profile:
        """COMPACT the whole profile past the total cap, then REGROUP each group still past a per-label cap.

        The per-label check reads COMPACT's result when COMPACT ran, so a group COMPACT left over its cap is
        regrouped in the same call instead of waiting for the next UPDATE.
        """
        profile = merged_profile
        total_items = len(profile.explicit_info) + len(profile.implicit_traits)
        if total_items > _PROFILE_MAX_ITEMS:
            logger.warning(
                "profile total cap breached (%d > %d items), running full compact",
                total_items,
                _PROFILE_MAX_ITEMS,
            )
            profile = await self._compact(
                profile,
                display_name=display_name,
                sender_id=sender_id,
                output_language=output_language,
                custom_instructions=custom_instructions,
                prompts=compact_prompts,
                available_categories=available_categories,
            )
        return await self._regroup_overcrowded(
            profile,
            display_name=display_name,
            sender_id=sender_id,
            output_language=output_language,
            custom_instructions=custom_instructions,
            prompts=regroup_prompts,
            available_categories=available_categories,
        )

    async def _regroup_overcrowded[ReplyT: _AnyRegroupOutput](
        self,
        profile: Profile,
        *,
        display_name: str,
        sender_id: str,
        prompts: _StagePrompts[ReplyT],
        custom_instructions: str | None,
        available_categories: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Profile:
        """REGROUP, once each, every group past its per-label cap or holding an item past the width backstop."""
        result = profile
        for bucket, label in _overcrowded_labels(
            profile.explicit_info, profile.implicit_traits
        ):
            result = await self._regroup(
                result,
                bucket=bucket,
                label=label,
                display_name=display_name,
                sender_id=sender_id,
                output_language=output_language,
                custom_instructions=custom_instructions,
                prompts=prompts,
                available_categories=available_categories,
            )
        return result

    async def _regroup[ReplyT: _AnyRegroupOutput](
        self,
        profile: Profile,
        *,
        bucket: str,
        label: str,
        display_name: str,
        sender_id: str,
        prompts: _StagePrompts[ReplyT],
        custom_instructions: str | None,
        available_categories: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Profile:
        """Reorganise one overcrowded group in place; every other item passes through untouched.

        A reply that still breaches the cap is accepted with a warning: regroup never loops.
        """
        wanted = _normalize(label)
        buckets: dict[str, Sequence[_Item]] = {
            "explicit_info": profile.explicit_info,
            "implicit_traits": profile.implicit_traits,
        }
        group = [item for item in buckets[bucket] if _in_group(item, wanted)]
        rest = [item for item in buckets[bucket] if not _in_group(item, wanted)]
        other_labels = sorted(
            {lbl for item in rest if (lbl := _item_label(item)) is not None}
        )
        logger.warning(
            "profile group over cap, regrouping: bucket=%s label_len=%d group_items=%d "
            "(cap %d/label, %d-unit width backstop)",
            bucket,
            len(label),
            len(group),
            _PROFILE_MAX_PER_CATEGORY,
            _ITEM_WIDTH_BACKSTOP,
        )
        messages = prompts.render(
            custom_instructions=custom_instructions,
            bucket=bucket,
            reply_skeleton=REGROUP_REPLY_SKELETONS[bucket],
            label=label,
            label_field=_LABEL_FIELDS[bucket],
            count=len(group),
            max_per_category=_PROFILE_MAX_PER_CATEGORY,
            other_labels=", ".join(other_labels) if other_labels else "(none)",
            items_text=_render_group(group),
            target_user=display_name,
            target_user_id=sender_id,
            available_categories=available_categories or "[]",
            language_rule=prompts.language_rule(output_language),
        )
        data = await _call_llm_for_profile_regroup(
            self._llm, messages, reply=prompts.reply, bucket=bucket
        )
        returned: list[dict[str, Any]] = data["items"]
        kept = [
            item for item in returned if _identity_key(item["description"]) is not None
        ]
        dropped_malformed = len(returned) - len(kept)
        if dropped_malformed:
            logger.warning(
                "profile regroup dropped %d malformed item(s) bucket=%s label_len=%d",
                dropped_malformed,
                bucket,
                len(label),
            )
        # A regrouped item has no traceable origin, so it takes the profile's own date.
        buckets[bucket] = _dedupe(
            [*rest, *_stored_items(bucket, kept, profile.timestamp)], source="regroup"
        )
        still = sum(1 for item in buckets[bucket] if _in_group(item, wanted))
        if still > _PROFILE_MAX_PER_CATEGORY:
            logger.warning(
                "profile regroup left label over cap: bucket=%s label_len=%d count=%d",
                bucket,
                len(label),
                still,
            )
        logger.info(
            "profile regroup done: bucket=%s label_len=%d group %d -> %d items",
            bucket,
            len(label),
            len(group),
            len(kept),
        )
        return _profile_from_buckets(
            owner_id=profile.owner_id,
            timestamp=profile.timestamp,
            explicit_info=buckets["explicit_info"],
            implicit_traits=buckets["implicit_traits"],
        )

    async def _compact[ReplyT: _AnyCompactOutput](
        self,
        profile: Profile,
        *,
        display_name: str,
        sender_id: str,
        prompts: _StagePrompts[ReplyT],
        custom_instructions: str | None,
        available_categories: str | None = None,
        output_language: OutputLanguage | str | None = None,
    ) -> Profile:
        total_items = len(profile.explicit_info) + len(profile.implicit_traits)
        messages = prompts.render(
            custom_instructions=custom_instructions,
            total_items=total_items,
            max_items=_PROFILE_MAX_ITEMS,
            max_per_category=_PROFILE_MAX_PER_CATEGORY,
            profile_text=_render_profile_for_compact(profile),
            target_user=display_name,
            target_user_id=sender_id,
            available_categories=available_categories or "[]",
            language_rule=prompts.language_rule(output_language),
        )

        data = await _call_llm_for_profile_compact(
            self._llm, messages, reply=prompts.reply
        )
        # A rewritten item cannot be traced to the ones it came from, so it takes the profile's own date.
        new_explicit = _dedupe(
            _stored_items("explicit_info", data["explicit_info"], profile.timestamp),
            source="compact",
        )
        new_implicit = _dedupe(
            _stored_items(
                "implicit_traits", data["implicit_traits"], profile.timestamp
            ),
            source="compact",
        )
        logger.info(
            "profile compacted: %d -> %d items",
            total_items,
            len(new_explicit) + len(new_implicit),
        )
        return _profile_from_buckets(
            owner_id=profile.owner_id,
            timestamp=profile.timestamp,
            explicit_info=new_explicit,
            implicit_traits=new_implicit,
        )
