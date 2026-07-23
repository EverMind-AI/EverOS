"""Map semantic agent-skill names to collision-safe path components.

Skill names remain authoritative human-readable values in frontmatter and
indexes. Only the filesystem component is encoded, with percent signs escaped
before path separators so literal escape-looking text cannot collide.
"""

from __future__ import annotations


def encode_skill_name_segment(skill_name: str) -> str:
    """Return one collision-safe path component for a semantic skill name."""
    return (
        skill_name.replace("%", "%25")
        .replace("/", "%2F")
        .replace("\\", "%5C")
    )
