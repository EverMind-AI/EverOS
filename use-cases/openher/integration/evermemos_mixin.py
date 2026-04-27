"""
EverMemosMixin — EverCore integration for ChatAgent.

This mixin handles all async memory operations in the ChatAgent lifecycle:
  Step 0:   Session context loading (first turn)
  Step 2.5: Relationship EMA (blend EverCore prior + LLM delta)
  Step 8.5: Collect async search results
  Step 11:  Fire-and-forget turn storage
  Step 12:  Async prefetch for next turn

The mixin pattern keeps EverCore concerns cleanly separated from the
core persona engine (drives, metabolism, neural network, style memory).

Full source: https://github.com/kellyvv/OpenHer/blob/main/agent/evermemos_mixin.py
"""

from __future__ import annotations

import asyncio


class EverMemosMixin:
    """EverCore async memory integration methods."""

    async def _evermemos_gather(self) -> dict:
        """
        Step 0: Load EverCore session context (first turn only).
        Subsequent turns reuse cached _session_ctx.
        Returns relationship_4d dict for GenomeEngine context.
        """
        empty_4d = {
            'relationship_depth': 0.0,
            'emotional_valence': 0.0,
            'trust_level': 0.0,
            'pending_foresight': 0.0,
        }

        if not (self.evermemos and self.evermemos.available):
            return empty_4d

        # Load once per session
        if self._turn_count == 1:
            self._session_ctx = await self.evermemos.load_session_context(
                user_id=self.evermemos_uid,
                persona_id=self.persona.persona_id,
                group_id=self._group_id,
            )
            if self._session_ctx.user_profile:
                self._user_profile = self._session_ctx.user_profile
            if self._session_ctx.episode_summary:
                self._episode_summary = self._session_ctx.episode_summary
            # Cache foresight text for Actor injection
            if self._session_ctx.foresight_text:
                self._foresight_text = self._session_ctx.foresight_text

        if not self._session_ctx:
            return empty_4d

        return self.evermemos.relationship_vector(self._session_ctx)

    def _apply_relationship_ema(
        self,
        prior: dict,
        rel_delta: dict,
        conversation_depth: float,
    ) -> dict:
        """
        Step 2.5: Semi-emergent relationship update.

        Pattern: posterior = clip(prior + LLM_delta) → EMA smooth
          alpha = clip(0.15 + 0.5 * depth, 0.15, 0.65)
          state_t = alpha * posterior + (1 - alpha) * state_{t-1}

        First turn initializes EMA state from prior, then applies delta normally.
        """
        # Map Critic output keys → context feature keys
        delta_map = {
            'relationship_depth': rel_delta.get('relationship_delta', 0.0),
            'emotional_valence': rel_delta.get('emotional_valence', 0.0),
            'trust_level': rel_delta.get('trust_delta', 0.0),
            'pending_foresight': 0.0,  # No delta for foresight (data-driven only)
        }

        # Initialize EMA on first turn
        if not self._relationship_ema:
            self._relationship_ema = dict(prior)

        # Compute posterior = clip(prior + delta)
        posterior = {}
        for k in prior:
            lo = -1.0 if k == 'emotional_valence' else 0.0
            posterior[k] = max(lo, min(1.0, prior[k] + delta_map.get(k, 0.0)))

        # Depth-modulated alpha: shallow → trust prior, deep → trust LLM
        alpha = max(0.15, min(0.65, 0.15 + 0.5 * conversation_depth))

        # EMA smooth
        ema = {}
        for k in prior:
            prev = self._relationship_ema.get(k, prior[k])
            ema[k] = round(alpha * posterior[k] + (1 - alpha) * prev, 4)
        self._relationship_ema = ema

        return ema

    def _evermemos_store_bg(self, user_message: str, reply: str) -> None:
        """Step 11: Fire-and-forget EverCore storage (asyncio.create_task)."""
        if not (self.evermemos and self.evermemos.available):
            return
        async def _do_store():
            try:
                await self.evermemos.store_turn(
                    user_id=self.evermemos_uid,
                    persona_id=self.persona.persona_id,
                    persona_name=self.persona.name,
                    user_name=self.user_name or "用户",
                    group_id=self._group_id,
                    user_message=user_message,
                    agent_reply=reply,
                )
            except Exception as e:
                print(f"  [evermemos] ❌ store failed: {type(e).__name__}: {e}")
        try:
            asyncio.create_task(_do_store())
        except Exception as e:
            print(f"  [evermemos] create_task error: {e}")

    def _evermemos_search_bg(self, user_message: str) -> None:
        """
        Step 12: Fire async RRF search for the current user_message.
        Results are collected at Step 8.5 of the NEXT turn.
        Cancels any pending search before starting a new one.
        """
        if not (self.evermemos and self.evermemos.available):
            return
        if not self._session_ctx or not self._session_ctx.has_history:
            return

        # Cancel any orphaned previous search task
        if self._search_task and not self._search_task.done():
            self._search_task.cancel()
            self._search_task = None

        try:
            self._search_turn_id = self._turn_count
            self._search_task = asyncio.create_task(
                self.evermemos.search_relevant_memories(
                    query=user_message,
                    user_id=self.evermemos_uid,
                    group_id=self._group_id,
                )
            )
        except Exception as e:
            print(f"  [evermemos] search create_task error: {e}")
            self._search_task = None

    async def _collect_search_results(self) -> None:
        """
        Collect previous turn's async search results (called at Step 8.5).
        Validates turn_id to prevent concurrent mismatch.
        Waits up to 0.5s; on timeout/error falls back to empty.
        """
        if self._search_task is None:
            return

        # Concurrency guard: reject stale results from wrong turn
        expected_turn = self._turn_count - 1
        if self._search_turn_id != expected_turn:
            self._search_task.cancel()
            self._search_task = None
            self._relevant_facts = ""
            self._relevant_episodes = ""
            self._relevant_profile = ""
            return

        try:
            facts, episodes, profile = await asyncio.wait_for(
                self._search_task, timeout=0.5
            )
            self._relevant_facts = facts
            self._relevant_episodes = episodes
            self._relevant_profile = profile
        except asyncio.TimeoutError:
            # Graceful degradation: use static session context
            self._relevant_facts = ""
            self._relevant_episodes = ""
            self._relevant_profile = ""
        except Exception:
            self._relevant_facts = ""
            self._relevant_episodes = ""
            self._relevant_profile = ""
        finally:
            self._search_task = None
