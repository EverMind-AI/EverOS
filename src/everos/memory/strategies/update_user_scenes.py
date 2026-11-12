"""update_user_scenes strategy -- a user's scenes, built from their episodes.

docs/scenes.md. The scene core (:mod:`._scenes`) over the user's own
episodes, ``users/<u>/scenes/``: this strategy names the owner and the
switch. It runs on the pipeline's episodes only (a Reflection merge is not a
new episode), with ``[scenes] users = true`` (off by default until its cost
is accepted) and an embedding provider.
"""

from __future__ import annotations

__all__ = ["UPDATE_USER_SCENES_SPEC", "update_user_scenes"]

from everos.config import load_settings
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted
from everos.memory.ports import Event, Immediate, Scope, StrategySpec

from ._scenes import update_scenes


def _applies(event: EpisodeExtracted, deps: MemoryDeps) -> bool:
    return (
        event.source == "pipeline"
        and deps.embedding is not None
        and load_settings().scenes.users
    )


async def update_user_scenes(event: EpisodeExtracted, deps: MemoryDeps) -> list[Event]:
    """Place the new episode in the user's scenes and rewrite the one it joins."""
    await update_scenes(
        deps,
        Scope(
            app_id=event.app_id,
            project_id=event.project_id,
            owner_id=event.owner_id,
            owner_type="user",
        ),
        entry_id=event.episode_entry_id,
        text=event.episode_text,
        timestamp_ms=event.episode_timestamp_ms,
        language=event.language,
        settings=load_settings(),
    )
    return []


UPDATE_USER_SCENES_SPEC = StrategySpec(
    name="update_user_scenes",
    fn=update_user_scenes,
    trigger=Immediate(on=[EpisodeExtracted]),
    applies_to=_applies,
    max_retries=2,
)
