"""Domain events produced by the memory pipelines and strategies.

Every event is an :class:`~everos.memory.ports.Event`: frozen, and
round-trippable through JSON, because the scheduler persists the triggering
event of each run and replays it after a crash. Fields added later carry
defaults so payloads persisted by an older release still validate.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from everos.memory.ports import Event


def group_key(app_id: str, project_id: str, group_id: str) -> str:
    """One string naming a group across spaces: ``<app>/<project>/<group>``."""
    return f"{app_id}/{project_id}/{group_id}"


def parse_group_key(key: str) -> tuple[str, str, str] | None:
    """``(app_id, project_id, group_id)`` back from :func:`group_key`."""
    parts = key.split("/")
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 and all(parts) else None


class EpisodeExtracted(Event):
    """Fired once per episode the conversation pipeline writes.

    A segment's episode lives in one space: a user's own (``owner_type``
    ``user``: a one-to-one conversation, ``owner_id`` the user) or a group's
    (``owner_type`` ``group``: a declared group, or the implicit group of a
    session's people, ``owner_id`` the group). The personal strategies run
    on a user's episodes only, the group strategies -- member-in-group
    profiles, the group profile, the group's scenes -- on a group's only
    (docs/group-memory.md, section 7.1 step 6): group content never feeds
    personal memory.

    Carries ``episode_text`` so the scene strategies can embed it without
    racing the cascade (cascade also embeds, but at the LanceDB layer —
    keeping a copy on the event is cheaper than polling LanceDB until the
    row appears). ``episode_timestamp_ms`` rides along so the scene cluster
    can stamp the algo-side ``Cluster.last_ts`` without a second md read.

    Attributes:
        speaker_ids: A group episode's human members whose own (role
            ``user``) messages are in the segment -- whose member-in-group
            profiles it can change. Empty on a user's episode.
        group_key: ``<app_id>/<project_id>/<group_id>`` on a group episode,
            filled from the three: the bucket an idle trigger keys on, so the
            tick it fires names the whole group (ids are path-safe, so the
            separator cannot occur inside one). ``None`` on a user's episode,
            which no idle trigger buckets.
    """

    memcell_id: str
    episode_entry_id: str
    episode_text: str
    episode_timestamp_ms: int
    owner_id: str
    session_id: str | None = None
    app_id: str = "default"
    project_id: str = "default"
    language: str | None = None
    """The episode's ``OutputLanguage`` value (see ``Episode.language``); the
    extractions this event drives write in it. ``None`` lets each extractor
    infer its own."""
    owner_type: Literal["user", "group"] = "user"
    speaker_ids: list[str] = Field(default_factory=list)
    group_key: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _drop_reflection_keys(cls, data: Any) -> Any:
        """Read a run an earlier 2.0 build persisted, before Reflection went.

        Its payload carries ``source`` (always ``"pipeline"`` for the runs of
        the strategies that remain; Reflection's own were ``finish_reflection``
        runs, dropped as an unknown strategy) and ``reflection`` (``null``).
        """
        if isinstance(data, dict) and ("source" in data or "reflection" in data):
            data = {k: v for k, v in data.items() if k not in _REFLECTION_KEYS}
        return data

    @model_validator(mode="after")
    def _fill_group_key(self) -> EpisodeExtracted:
        if self.owner_type == "group" and self.group_key is None:
            key = group_key(self.app_id, self.project_id, self.owner_id)
            object.__setattr__(self, "group_key", key)
        return self


_REFLECTION_KEYS = frozenset({"source", "reflection"})


class GroupFlushed(Event):
    """Fired when a flush of a session bound to a group has run.

    The member-in-group profiles waiting on ``[groups] member_profile_every``
    are brought up to date on it (docs/group-memory.md, section 7.4): episodes
    are searchable at once, and a flush is where a client says it wants the
    rest current too. Fired whether or not the flush cut a segment.
    """

    group_id: str
    session_id: str
    app_id: str = "default"
    project_id: str = "default"
