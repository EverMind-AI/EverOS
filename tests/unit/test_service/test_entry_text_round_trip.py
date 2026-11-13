"""A memory's text survives its entry whatever it holds (E-1).

A body line that starts a section (``### Day 2``) or an entry marker in the
text used to end the entry's section -- or the entry -- early on read: the
index, ``/memory/history`` and every later rewrite saw only the text before
it, while the markdown kept it all. Such text is now written behind one more
backslash and read back as it was.
"""

from __future__ import annotations

from pathlib import Path

from everos.service import GetRequest
from tests.helpers.groups import APP, PROJECT, GroupWorld

T0 = 1_772_323_200_000

TEXTS = {
    "h3": "Amy planned the trip.\n### Day 2\nShe booked the Alfama hotel.",
    "h3_alone": "Amy planned the trip.\n###\nDay 2 was free.",
    "close_marker": "Trip <!-- /entry:ep_20260301_00000001 --> booked.",
    "open_marker": "Trip <!-- entry:ep_20260301_00000009 --> booked.",
    "escaped_already": "A path \\### and a line:\n\\### stays as typed",
}


async def test_section_lines_and_markers_round_trip(tmp_path: Path) -> None:
    async with GroupWorld.open(tmp_path) as world:
        ids = {
            name: await world.personal_episode("amy", text, at=T0 + i)
            for i, (name, text) in enumerate(TEXTS.items())
        }
        await world.project()
        got = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                user_id="amy",
                memory_type="episode",
                page_size=50,
            )
        )
        indexed = {e.id: e.episode for e in got.data.episodes}
        for name, text in TEXTS.items():
            assert indexed[ids[name]] == text, name
            history = await world.memory.memory_history(
                app_id=APP, project_id=PROJECT, record_id=ids[name], user_id="amy"
            )
            shown = history.versions[0].content
            assert shown["content"] == text, name
            # The helper's summary and subject cut the text: one ends inside it.
            assert (shown["summary"], shown["subject"]) == (
                text[:40].rstrip(),
                text[:30].rstrip(),
            )
        log = next(world.root.root.glob("*/*/users/amy/episodes/episode-*.md"))
        raw = log.read_text(encoding="utf-8")
        assert "\n\\### Day 2\n" in raw
        assert "<!-- \\/entry:ep_20260301_00000001 -->" in raw


async def test_an_entry_written_before_reads_unchanged(tmp_path: Path) -> None:
    """A store's entries hold no escape: a body reads exactly as before."""
    async with GroupWorld.open(tmp_path) as world:
        plain = "Lisbon tram at 9:40. **Note**: bring cash; #5 # tags ## two"
        record = await world.personal_episode("amy", plain, at=T0)
        log = next(world.root.root.glob("*/*/users/amy/episodes/episode-*.md"))
        assert plain in log.read_text(encoding="utf-8")
        await world.project()
        history = await world.memory.memory_history(
            app_id=APP, project_id=PROJECT, record_id=record, user_id="amy"
        )
        assert history.versions[0].content["content"] == plain
