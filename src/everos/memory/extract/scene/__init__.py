"""Scenes: the narrative call, the version it writes, how forgetting reaches it.

A scene is an owner's episodes about one matter -- a user's or a group's --
with one narrative written over them, ending on where things stand
(docs/scenes.md; a group's, docs/group-memory.md, section 7.7). The narrative
is one capped call each time episodes join the scene (``chat.scene_update``,
:mod:`.prompt`, made through :class:`SceneNarrator`); what it reads is each
member's current version (:mod:`.members`); a version's span, attribution
and origin come from its members (:mod:`.version`); a user's forget, its undo
and a deletion reach the scene through its episodes (:mod:`.forget`), and a
forget inside a group writes the scene again through the same narrator. The
prompt lives in EverOS until an everalgo operator takes over the same
registry id.

External usage::

    from everos.memory.extract.scene import SceneNarrator, scene_version
"""

from __future__ import annotations

from .forget import forget_orphaned_scenes as forget_orphaned_scenes
from .forget import newest_versions as newest_versions
from .forget import remove_scenes_naming as remove_scenes_naming
from .forget import restore_revived_scenes as restore_revived_scenes
from .members import alive_members as alive_members
from .members import current_versions as current_versions
from .narrate import SceneCaps as SceneCaps
from .narrate import SceneNarrator as SceneNarrator
from .narrate import SceneText as SceneText
from .narrate import write_narrative as write_narrative
from .prompt import SCENE_UPDATE as SCENE_UPDATE
from .prompt import SceneReply as SceneReply
from .prompt import build_messages as build_messages
from .prompt import clip_narrative as clip_narrative
from .prompt import narrative_char_cap as narrative_char_cap
from .prompt import reply_token_cap as reply_token_cap
from .version import scene_cluster_kind as scene_cluster_kind
from .version import scene_version as scene_version

__all__ = [
    "SCENE_UPDATE",
    "SceneCaps",
    "SceneNarrator",
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
    "scene_cluster_kind",
    "scene_version",
    "write_narrative",
]
