"""Parse EverOS memory markdown files into entry records.

EverOS persists memory as human-readable markdown (the single source of
truth). Daily-log files (episodes, atomic facts, foresights, agent cases)
hold entry blocks delimited by ``<!-- entry:<eid> -->`` markers, each with
inline ``**field**: value`` lines and ``### Subject / ### Summary /
### Content`` sections. Single-file artifacts (``user.md``, ``agent.md``,
skill files) carry no entry markers and are treated as one whole-file
artifact instead.

This module is a pure, dependency-free parser used by the MemTrace
file-tracking tracer (:mod:`everos.infra.memtrace.tracker`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_ENTRY_OPEN = re.compile(r"<!--\s*entry:([^\s>]+?)\s*-->")
_ENTRY_CLOSE = re.compile(r"<!--\s*/entry:([^\s>]+?)\s*-->")
_INLINE_FIELD = re.compile(r"^\*\*([A-Za-z_0-9]+)\*\*:\s*(.*)$")
_SECTION_HEAD = re.compile(r"^###\s+(.+?)\s*$")
_FRONTMATTER_LINE = re.compile(r"^([A-Za-z_0-9]+):\s*(.*)$")

# Daily-log kinds whose writer emits kind-specific body sections instead of
# the generic ``### Content`` (atomic facts: ``### Fact``; foresights:
# ``### Foresight`` / ``### Evidence``; agent cases: ``### TaskIntent`` /
# ``### Approach`` / ``### KeyInsight``). Their sections are folded into
# ``content`` (in writer order) so the tracker's entry snapshot carries the
# entry body text; a generic ``### Content`` section, when present, wins.
_BODY_SECTIONS_BY_KIND: dict[str, tuple[str, ...]] = {
    "atomic_fact": ("fact",),
    "foresight": ("foresight", "evidence"),
    "agent_case": ("taskintent", "approach", "keyinsight"),
}

# Path components that identify the memory kind of a markdown file.
_KIND_BY_DIR = {
    "episodes": "episode",
    ".atomic_facts": "atomic_fact",
    ".foresights": "foresight",
    ".cases": "agent_case",
    "knowledge": "knowledge",
}

_DEFAULT_APP_DIR = "default_app"
_DEFAULT_PROJECT_DIR = "default_project"


@dataclass
class MemoryFileEntry:
    """One ``<!-- entry:... -->`` block parsed from a memory markdown file."""

    entry_id: str
    app_id: str
    project_id: str
    owner_id: str | None
    session_id: str | None
    timestamp: str | None
    parent_type: str | None
    parent_id: str | None
    sender_ids: list[str] = field(default_factory=list)
    subject: str = ""
    summary: str = ""
    content: str = ""
    fields: dict[str, str] = field(default_factory=dict)
    path: str = ""
    kind: str = "memory_entry"


@dataclass
class MemoryFileSnapshot:
    """Parsed view of one memory markdown file."""

    path: str
    kind: str
    app_id: str
    project_id: str
    owner_id: str | None
    frontmatter: dict[str, str]
    entries: list[MemoryFileEntry]
    body: str


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Naively parse the leading YAML frontmatter block (flat keys only)."""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    frontmatter: dict[str, str] = {}
    for line in text[3:end].splitlines():
        m = _FRONTMATTER_LINE.match(line.strip())
        if m:
            frontmatter[m.group(1)] = m.group(2).strip().strip("'\"")
    return frontmatter


def _parse_sender_ids(raw: str) -> list[str]:
    """Parse an inline ``sender_ids`` value such as ``[alice, bob]``."""
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    return [part.strip() for part in raw.split(",") if part.strip()]


def _kind_from_path(path: Path) -> str:
    """Derive the memory kind from the file's path components."""
    parts = set(path.parts)
    if "skills" in parts:
        # Only the main file represents the skill memory unit. Reference
        # markdown files are attachments, not independent AgentSkills.
        return "agent_skill" if path.name == "SKILL.md" else "skill_reference"
    for part, kind in _KIND_BY_DIR.items():
        if part in parts:
            return kind
    if path.name == "user.md":
        return "profile"
    if path.name == "agent.md":
        return "agent_profile"
    return "memory_file"


def _owner_from_path(path: Path) -> str | None:
    """Recover the owner id from a ``users/<id>/`` or ``agents/<id>/`` segment."""
    parts = path.parts
    for marker in ("users", "agents"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return None


def _scope_from_path(path: Path) -> tuple[str, str]:
    """Recover the app/project scope preceding a business root directory."""
    parts = path.parts
    for marker in ("users", "agents", "knowledge"):
        if marker not in parts:
            continue
        idx = parts.index(marker)
        if idx < 2:
            break
        app_dir, project_dir = parts[idx - 2], parts[idx - 1]
        app_id = "default" if app_dir == _DEFAULT_APP_DIR else app_dir
        project_id = "default" if project_dir == _DEFAULT_PROJECT_DIR else project_dir
        return app_id, project_id
    # Legacy/unscoped test data and hand-written memory roots belong to the
    # default business partition.
    return "default", "default"


def _parse_entry_block(
    entry_id: str,
    block: str,
    *,
    path: str,
    kind: str,
    app_id: str,
    project_id: str,
    default_owner: str | None,
) -> MemoryFileEntry:
    """Parse one entry block body (between the entry markers)."""
    fields: dict[str, str] = {}
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for line in block.splitlines():
        field_match = _INLINE_FIELD.match(line.strip())
        section_match = _SECTION_HEAD.match(line.strip())
        if section_match:
            current_section = section_match.group(1).strip().lower()
            sections.setdefault(current_section, [])
        elif field_match and current_section is None:
            fields[field_match.group(1)] = field_match.group(2).strip()
        elif current_section is not None:
            sections[current_section].append(line)
    subject = "\n".join(sections.get("subject", [])).strip()
    summary = "\n".join(sections.get("summary", [])).strip()
    content = "\n".join(sections.get("content", [])).strip()
    if not content and kind in _BODY_SECTIONS_BY_KIND:
        content = "\n\n".join(
            body
            for body in (
                "\n".join(sections.get(name, [])).strip()
                for name in _BODY_SECTIONS_BY_KIND[kind]
            )
            if body
        )
    return MemoryFileEntry(
        entry_id=entry_id,
        app_id=app_id,
        project_id=project_id,
        owner_id=fields.get("owner_id") or default_owner,
        session_id=fields.get("session_id") or None,
        timestamp=fields.get("timestamp") or None,
        parent_type=fields.get("parent_type") or None,
        parent_id=fields.get("parent_id") or None,
        sender_ids=_parse_sender_ids(fields.get("sender_ids", "")),
        subject=subject,
        summary=summary,
        content=content,
        fields=fields,
        path=path,
        kind=kind,
    )


def parse_memory_file(
    path: Path,
    *,
    relative_to: Path | None = None,
) -> MemoryFileSnapshot:
    """Parse one memory markdown file into a snapshot.

    Args:
        path (`Path`):
            The markdown file to parse.
        relative_to (`Path | None`, optional):
            Root used to compute the relative path stored on the snapshot.

    Returns:
        `MemoryFileSnapshot`:
            The parsed file: frontmatter, entry blocks (if any) and the
            raw body.
    """
    text = path.read_text(encoding="utf-8")
    rel = str(path.relative_to(relative_to)) if relative_to else str(path)
    kind = _kind_from_path(path)
    frontmatter = _parse_frontmatter(text)
    app_id, project_id = _scope_from_path(Path(rel))
    owner = (
        frontmatter.get("user_id")
        or frontmatter.get("agent_id")
        or _owner_from_path(path)
    )

    entries: list[MemoryFileEntry] = []
    opens = list(_ENTRY_OPEN.finditer(text))
    for i, open_match in enumerate(opens):
        entry_id = open_match.group(1)
        close_match = _ENTRY_CLOSE.search(text, open_match.end())
        block_end = (
            close_match.start()
            if close_match
            else (opens[i + 1].start() if i + 1 < len(opens) else len(text))
        )
        block = text[open_match.end() : block_end]
        entries.append(
            _parse_entry_block(
                entry_id,
                block,
                path=rel,
                kind=kind,
                app_id=app_id,
                project_id=project_id,
                default_owner=owner,
            )
        )

    return MemoryFileSnapshot(
        path=rel,
        kind=kind,
        app_id=app_id,
        project_id=project_id,
        owner_id=owner,
        frontmatter=frontmatter,
        entries=entries,
        body=text,
    )
