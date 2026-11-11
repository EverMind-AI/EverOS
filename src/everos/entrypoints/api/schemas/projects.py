"""Wire shapes of the project reads (docs/group-memory.md, sections 5.1 and 8.7)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from everos.service import GroupProfileEntry


class ProjectGroupDTO(BaseModel):
    """One group of the project the asker is in."""

    model_config = ConfigDict(extra="forbid")

    group_id: str
    name: str = ""


class ProjectMemberDTO(BaseModel):
    """A current member of the asker's groups, with the groups they share."""

    model_config = ConfigDict(extra="forbid")

    member_id: str
    display_name: str
    kind: str
    group_ids: list[str] = Field(default_factory=list)


class ProjectDTO(BaseModel):
    """``GET /projects/{project_id}``: the asker's groups and their members."""

    model_config = ConfigDict(extra="forbid")

    app_id: str
    project_id: str
    groups: list[ProjectGroupDTO] = Field(default_factory=list)
    members: list[ProjectMemberDTO] = Field(default_factory=list)


class ProjectProfileDTO(BaseModel):
    """``GET /projects/{project_id}/profile``: the visible current items.

    Each item names its group (``group_id``); oldest first. The same decision
    taken in two groups is two items.
    """

    model_config = ConfigDict(extra="forbid")

    app_id: str
    project_id: str
    group_ids: list[str] = Field(default_factory=list)
    items: list[GroupProfileEntry] = Field(default_factory=list)
