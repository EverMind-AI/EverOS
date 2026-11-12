"""update_group_scenes strategy -- a group's scenes, built from its episodes.

docs/group-memory.md, section 7.7. The scene core (:mod:`._scenes`) a
user's scenes run, over the group's episodes (a declared group's, or the
implicit group of a session's people), ``groups/<g>/scenes/``: this strategy
names the owner and the switch. A version spans its members
(``source_start`` of the first to the end of the last), so a read shows it
only to an asker who can see every one of them; the episodes still return on
their own. Runs with ``[scenes] groups = true`` (off by default until its
cost is accepted) and an embedding provider.
"""

from __future__ import annotations

__all__ = ["UPDATE_GROUP_SCENES_SPEC", "update_group_scenes"]

from everos.config import load_settings
from everos.memory.deps import MemoryDeps
from everos.memory.events import EpisodeExtracted
from everos.memory.ports import Event, Immediate, Scope, StrategySpec

from ._scenes import update_scenes


def _applies(event: EpisodeExtracted, deps: MemoryDeps) -> bool:
    return (
        event.owner_type == "group"
        and deps.embedding is not None
        and load_settings().scenes.groups
    )


async def update_group_scenes(event: EpisodeExtracted, deps: MemoryDeps) -> list[Event]:
    """Place the new episode in the group's scenes and rewrite the one it joins."""
    await update_scenes(
        deps,
        Scope(
            app_id=event.app_id,
            project_id=event.project_id,
            owner_id=event.owner_id,
            owner_type="group",
        ),
        entry_id=event.episode_entry_id,
        text=event.episode_text,
        timestamp_ms=event.episode_timestamp_ms,
        language=event.language,
        settings=load_settings(),
    )
    return []


UPDATE_GROUP_SCENES_SPEC = StrategySpec(
    name="update_group_scenes",
    fn=update_group_scenes,
    trigger=Immediate(on=[EpisodeExtracted]),
    applies_to=_applies,
    max_retries=2,
)
