"""A user's scenes: the narrative call and how forgetting reaches it (docs/scenes.md).

A scene is a user's episodes about one matter with one narrative written over
them, ending on where things stand. The narrative is one capped call each
time episodes join the scene (``chat.scene_update``, :mod:`.prompt`); what it
reads is each member's current version (:mod:`.members`); a forget, its undo
and a deletion reach the scene through its episodes (:mod:`.forget`). The
prompt lives in EverOS until an everalgo operator takes over the same
registry id.

External usage::

    from everos.memory.extract.scene import SceneCaps, write_narrative
"""

from __future__ import annotations

from .forget import forget_orphaned_scenes as forget_orphaned_scenes
from .forget import newest_versions as newest_versions
from .forget import remove_scenes_naming as remove_scenes_naming
from .forget import restore_revived_scenes as restore_revived_scenes
from .members import alive_members as alive_members
from .members import current_versions as current_versions
from .narrate import SceneCaps as SceneCaps
from .narrate import SceneText as SceneText
from .narrate import write_narrative as write_narrative
from .prompt import SCENE_UPDATE as SCENE_UPDATE
from .prompt import SceneReply as SceneReply
from .prompt import build_messages as build_messages
from .prompt import clip_narrative as clip_narrative
from .prompt import narrative_char_cap as narrative_char_cap
from .prompt import reply_token_cap as reply_token_cap

__all__ = [
    "SCENE_UPDATE",
    "SceneCaps",
    "SceneReply",
    "SceneText",
    "alive_members",
    "build_messages",
    "clip_narrative",
    "current_versions",
    "forget_orphaned_scenes",
    "narrative_char_cap",
    "newest_versions",
    "remove_scenes_naming",
    "reply_token_cap",
    "restore_revived_scenes",
    "write_narrative",
]
