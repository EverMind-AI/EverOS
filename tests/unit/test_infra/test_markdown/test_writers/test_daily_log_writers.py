"""The frontmatter timestamp contract the daily-log writers share.

The daily-log kinds (episode / image memory / scene / group profile) all share
``BaseDailyWriter`` plumbing — exhaustive chassis tests live in
``test_base.py`` and ``test_episode_writer.py`` indirectly via the e2e
flows. Here we focus on the frontmatter shape each subclass owns through
``_frontmatter_updates``, on a fresh tmp memory_root.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import pytest

from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.markdown import (
    EpisodeWriter,
    GroupProfileWriter,
    MarkdownReader,
)


@pytest.fixture
def memory_root(tmp_path: Path) -> MemoryRoot:
    mr = MemoryRoot(tmp_path)
    mr.ensure()
    return mr


# ── Display-tz contract for frontmatter timestamps (Gap #5) ────────────


async def test_daily_log_frontmatter_last_appended_at_carries_display_tz_offset(
    memory_root: MemoryRoot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``last_appended_at`` in markdown frontmatter renders in the display tz.

    Markdown frontmatter is a display-side artefact (users read the file
    directly), so ``last_appended_at`` must use
    :func:`get_now_with_timezone` not :func:`get_utc_now`. Pins that
    contract end-to-end: configure ``EVEROS_MEMORY__TIMEZONE=Asia/Shanghai``,
    write an entry, read the .md file, assert the literal string ends
    with ``+08:00``.

    Checks a user-track writer (``EpisodeWriter``) and the group-track
    ``GroupProfileWriter`` — they share ``BaseDailyWriter`` plumbing so a
    regression on one would likely affect both, but pinning each rules
    out per-subclass shadowing of ``_frontmatter_updates``.
    """
    from everos.config import load_settings
    from everos.core.utils import datetime as _dt_module

    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "Asia/Shanghai")
    load_settings.cache_clear()
    _dt_module._display_tz.cache_clear()

    today = _dt.date(2026, 5, 15)

    # Episode
    ep_writer = EpisodeWriter(memory_root)
    await ep_writer.append_entry(
        "u1",
        inline={
            "owner_id": "u1",
            "session_id": "s1",
            "timestamp": "2026-05-15T10:00:00+00:00",
            "parent_id": "mc_1",
            "sender_ids": ["u1"],
        },
        sections={"Summary": "x", "Content": "x"},
        date=today,
    )
    ep_path = memory_root.users_dir() / "u1" / "episodes" / "episode-2026-05-15.md"
    ep_fm = (await MarkdownReader.read(ep_path)).frontmatter
    assert ep_fm["last_appended_at"].endswith("+08:00"), ep_fm["last_appended_at"]

    # Group profile
    gp_writer = GroupProfileWriter(memory_root)
    await gp_writer.append_entry(
        "trip",
        inline={"kind": "decision", "observed_at": "2026-05-15T10:00:00+00:00"},
        sections={"Statement": "x"},
        date=today,
    )
    gp_path = (
        memory_root.groups_dir() / "trip" / "profile" / "group_profile-2026-05-15.md"
    )
    gp_fm = (await MarkdownReader.read(gp_path)).frontmatter
    assert gp_fm["last_appended_at"].endswith("+08:00"), gp_fm["last_appended_at"]
