"""The member profile's source check (docs/group-memory.md, 7.4).

``own_segments`` keeps only the segments with a message of the member's own:
no item may rest on others' words alone. The items' provenance, versions and
history are the shared profile writer's (``test_profile_items.py``).
"""

from __future__ import annotations

from everalgo.types import ChatMessage, MemCell

from everos.memory.extract.pipeline.member_profile import own_segments


def test_only_segments_with_the_members_own_messages_are_read() -> None:
    def cell(*senders: tuple[str, str]) -> MemCell:
        items = [
            ChatMessage(
                id=f"m{i}", role=role, sender_id=s, content="x", timestamp=i + 1
            )
            for i, (s, role) in enumerate(senders)
        ]
        return MemCell(items=items, timestamp=len(items))

    mine = cell(("u_betty", "user"), ("u_chen", "user"))
    others = cell(("u_chen", "user"), ("bot_ava", "assistant"))
    assert own_segments([mine, others], "u_betty") == [mine]
