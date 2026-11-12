"""Forget inside a group, end to end over real markdown, system.db and LanceDB.

docs/group-memory.md, sections 5.5 and 9.4, task 12. A group chat goes
through the bootstrap's memory service (add, flush), the group strategies
write the member-in-group profiles and the group profile, and the cascade
projects everything into the index; then a message is forgotten. What these
pin:

- a message forgotten by its sender re-extracts its segment, and its text is
  nowhere afterwards: not in any markdown file under the memory root, not in
  an index row, not in a search result for any member, not in a member
  profile, not in a group-profile item -- the stale items written again from
  their remaining sources by the next update;
- ``dry_run`` reports exactly what the real run then touches;
- another member forgetting someone else's message is 403; an owner or an
  admin may forget any message; a former member and a stranger are 403;
- undo puts the previous versions back exactly (markdown and what search
  returns);
- a member who leaves stays visible to the others: leaving is not a forget;
- a member, a session, records and a picture are forgotten along the same
  rules, and a group-scoped delete is an owner's or admin's forget.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient

from everos.core.errors import ConflictError, ForbiddenError, NotFoundError
from everos.memory.models import Cluster, Scene
from everos.memory.ports import NoSourceCheck
from everos.memory.strategies import STRATEGY_SPECS
from everos.service import GetRequest, GroupForgetTarget, SearchRequest
from tests.helpers.groups import APP, PROJECT, GroupWorld, join, leave

T0 = 1_772_323_200_000
SECRET = "ZEBRA4417"
"""The forgotten message's own words: every derived record echoes it."""
MARKERS = (SECRET, "LISBONVENUE", "FRIDAYOK", "HOTELBOOKED")

_GP_LINE = re.compile(
    r"^\[([^\]]+)\] \d{4}-\d{2}-\d{2} \d{2}:\d{2} ([^:\n]+): (.*)$", re.MULTILINE
)
_GP_SEGMENT = re.compile(r"^=== SEGMENT (\S+) ===$", re.MULTILINE)
_TARGET = re.compile(r"The target user is (.+?), the speaker")


def _prompt(messages: Any) -> str:
    return "\n".join(str(getattr(m, "content", m)) for m in messages)


def _markers(text: str) -> list[str]:
    return [m for m in MARKERS if m in text]


class _Model:
    """A deterministic model whose every record repeats what it was shown."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.down = False
        """Every call fails while set: a model that is down."""
        self.failing: set[str] = set()
        """Replies (by response model name) that fail."""

    def __call__(self, messages: Any, *, response_format: type, **_: Any) -> str:
        name = response_format.__name__
        prompt = _prompt(messages)
        self.calls.append(name)
        if self.down or name in self.failing:
            raise RuntimeError("model down")
        if name == "_BatchReply":
            return json.dumps(
                {"reasoning": "s", "boundaries": [], "tail_complete": False}
            )
        if name == "_EpisodeReply":
            said = ", ".join(_markers(prompt)) or "nothing"
            return json.dumps(
                {
                    "title": "Trip plans",
                    "summary": f"Plans: {said}.",
                    "content": f"The group talked about {said}.",
                }
            )
        if name == "_ProfileUpdateReply":
            found = _TARGET.search(prompt)
            who = found.group(1).strip() if found else "?"
            said = ", ".join(_markers(prompt))
            item = {
                "evidence": f"{who} said {said}",
                "description": f"{who} cares about {said}",
                "category": "plans",
            }
            return json.dumps(
                {
                    "operations": [
                        {"action": "add", "type": "explicit_info", "data": item}
                    ]
                }
            )
        if name == "GroupProfileReply":
            return json.dumps({"operations": _profile_operations(prompt)})
        if name == "_EpisodeReflectReply" or "Reflect" in name:
            said = ", ".join(_markers(prompt)) or "nothing"
            return json.dumps(
                {"title": "Scene", "summary": said, "content": f"Scene of {said}."}
            )
        raise AssertionError(f"unexpected model call {name}")


_BLANK: dict[str, Any] = {
    "target": "",
    "topic_id": "",
    "name": "",
    "summary": "",
    "status": "",
    "statement": "",
    "decided_by": [],
    "agreed_by": [],
    "assignees": [],
    "applies_to": [],
    "held_by": [],
    "about": "",
    "member_id": "",
    "role": "",
    "confidence": "",
}


def _profile_operations(prompt: str) -> list[dict[str, Any]]:
    """A decision citing every message of the batch, a topic citing the first."""
    body = prompt[prompt.find("=== SEGMENT") :]
    evidence: list[dict[str, str]] = []
    segments = _GP_SEGMENT.findall(body)
    texts = _GP_SEGMENT.split(body)[2::2]
    for segment, text in zip(segments, texts, strict=False):
        for message_id, _speaker, said in _GP_LINE.findall(text):
            evidence.append({"segment": segment, "message": message_id, "quote": said})
    if not evidence:
        return []
    quotes = " / ".join(e["quote"] for e in evidence)
    return [
        {
            **_BLANK,
            "op": "decision.new",
            "statement": f"The plan: {quotes}",
            "evidence": evidence,
        },
        {
            **_BLANK,
            "op": "topic.new",
            "name": f"First words: {evidence[0]['quote']}",
            "summary": "s",
            "status": "exploring",
            "evidence": evidence[:1],
        },
    ]


@dataclass
class _Chat:
    world: GroupWorld
    model: _Model


ROSTER = [
    join("u_chen", T0, role="owner", display_name="Chen"),
    join("u_evan", T0, role="admin", display_name="Evan"),
    join("u_betty", T0, display_name="Betty"),
    join("u_dan", T0, display_name="Dan"),
    join("u_gone", T0, display_name="Gone"),
    leave("u_gone", T0 + 500),
]

MESSAGES = [
    ("m-1", "u_betty", f"My locker code is {SECRET}, keep it to yourselves", 1_000),
    ("m-2", "u_chen", "We meet at the LISBONVENUE on Friday", 2_000),
    ("m-3", "u_betty", "FRIDAYOK for me", 3_000),
    ("m-4", "u_dan", "HOTELBOOKED for all of us", 4_000),
]


async def _add(
    world: GroupWorld, messages: Sequence[tuple[str, str, str, int]]
) -> None:
    await world.memory.add(
        {
            "session_id": "s_trip",
            "app_id": APP,
            "project_id": PROJECT,
            "group_id": "trip",
            "messages": [
                {
                    "message_id": mid,
                    "sender_id": who,
                    "role": "user",
                    "timestamp": T0 + at,
                    "content": text,
                }
                for mid, who, text, at in messages
            ],
        },
        request_id="r",
        async_mode=False,
        defer_extraction=True,
    )


async def _flush(world: GroupWorld) -> None:
    await world.memory.flush(
        session_id="s_trip", request_id="f", app_id=APP, project_id=PROJECT
    )
    await world.run(*STRATEGY_SPECS)
    await world.project()


@pytest.fixture
async def chat(tmp_path: Path) -> AsyncIterator[_Chat]:
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        await world.roster("trip", ROSTER)
        await _add(world, MESSAGES)
        await _flush(world)
        yield _Chat(world=world, model=model)


def _markdown(world: GroupWorld) -> dict[str, str]:
    return {
        p.relative_to(world.root.root).as_posix(): p.read_text("utf-8")
        for p in world.root.root.rglob("*.md")
        if not p.relative_to(world.root.root).parts[0].startswith(".")
    }


async def _index_rows(world: GroupWorld) -> str:
    rows: list[str] = []
    for repo in world.storage.repos.all():
        try:
            found = await repo.scan()
        except Exception:  # a table nothing was ever written to
            continue
        rows += [json.dumps(r.model_dump(mode="json"), default=str) for r in found]
    return "\n".join(rows)


async def _search_text(world: GroupWorld, asker: str) -> str:
    response = await world.memory.search(
        SearchRequest(
            app_id=APP,
            project_id=PROJECT,
            query="locker code LISBONVENUE FRIDAYOK plans",
            top_k=20,
            group_id="trip",
            asker_id=asker,
            include_profile=True,
        )
    )
    return json.dumps(response.data.model_dump(mode="json"))


async def _gets(world: GroupWorld, asker: str) -> str:
    blobs: list[str] = []
    for memory_type, extra in (
        ("episode", {}),
        ("image_memory", {}),
        ("scene", {}),
        ("group_profile", {"include_superseded": True}),
        ("member_profile", {"member_id": "u_betty"}),
        ("member_profile", {"member_id": "u_chen"}),
    ):
        response = await world.memory.get(
            GetRequest(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id=asker,
                memory_type=memory_type,
                page_size=100,
                **extra,
            )
        )
        blobs.append(json.dumps(response.model_dump(mode="json")))
    return "\n".join(blobs)


def _message(*message_ids: str) -> GroupForgetTarget:
    return GroupForgetTarget(
        kind="message", session_id="s_trip", message_ids=tuple(message_ids)
    )


async def _forget(
    world: GroupWorld, asker: str, target: GroupForgetTarget, **kw: Any
) -> Any:
    return await world.memory.forget_in_group(
        app_id=APP,
        project_id=PROJECT,
        group_id="trip",
        asker_id=asker,
        target=target,
        **kw,
    )


def _plan_view(result: Any) -> dict[str, Any]:
    """What a dry run and the run that follows it must agree on."""
    return {
        "messages": result.messages,
        "episodes": result.episodes,
        "rewritten": result.rewritten,
        "image_memories": result.image_memories,
        "images": result.images,
        "scenes": result.scenes,
        "rewritten_scenes": result.rewritten_scenes,
        "member_profiles": result.member_profiles,
        "member_profile_items": len(result.member_profile_items),
        "stale_member_profile_items": len(result.stale_member_profile_items),
        "group_profile_items": result.group_profile_items,
        "stale_group_profile_items": result.stale_group_profile_items,
    }


# ── the acceptance case ──────────────────────────────────────────────────


async def test_a_message_forgotten_by_its_sender_is_nowhere_afterwards(
    chat: _Chat,
) -> None:
    world = chat.world
    # Every kind of record holds the secret before the forget.
    before = _markdown(world)
    assert any(SECRET in text for path, text in before.items() if "/episodes/" in path)
    assert any(SECRET in text for path, text in before.items() if "/members/" in path)
    assert any(SECRET in text for path, text in before.items() if "/profile/" in path)
    assert SECRET in await _index_rows(world)
    assert SECRET in await _search_text(world, "u_chen")

    dry = await _forget(world, "u_betty", _message("m-1"), dry_run=True)
    assert dry.dry_run and dry.forget_id == ""
    assert _markdown(world) == before  # a dry run touches nothing
    real = await _forget(world, "u_betty", _message("m-1"), reason="mistake")

    assert real.forget_id.startswith("fg_")
    assert _plan_view(real) == _plan_view(dry)
    assert real.messages == (("s_trip", "m-1"),)
    assert real.rewritten == 1 and len(real.episodes) == 1
    assert len(real.new_episodes) == 1
    assert real.group_profile_items and real.stale_group_profile_items
    assert real.stale_member_profile_items  # Betty and Chen spoke in the segment
    # The next update writes the stale items again from what is left.
    await _flush(world)

    after = _markdown(world)
    leaks = [path for path, text in after.items() if SECRET in text]
    assert leaks == []
    assert SECRET not in await _index_rows(world)
    for asker in ("u_chen", "u_evan", "u_betty", "u_dan"):
        assert SECRET not in await _search_text(world, asker), asker
        assert SECRET not in await _gets(world, asker), asker
    # What was not forgotten is still remembered, rewritten.
    assert "LISBONVENUE" in await _search_text(world, "u_dan")
    betty = await world.stores.member_profile.read(world.scope("trip"), "u_betty")
    assert betty is not None and betty.profile.explicit_info  # rewritten
    items = await world.stores.group_profile.items(world.scope("trip"))
    assert any("LISBONVENUE" in i.item.statement for i in items)
    # The segment itself keeps the message, for an undo.
    cells = await world.stores.memcell.cells_in_session(world.session("s_trip"))
    assert any(SECRET in str(i.content) for c in cells.values() for i in c.items)


async def test_undo_restores_the_previous_versions(chat: _Chat) -> None:
    world = chat.world
    before = _markdown(world)
    episodes_before = {p: t for p, t in before.items() if "/episodes/" in p}
    profiles_before = {
        m: await world.stores.member_profile.read(world.scope("trip"), m)
        for m in ("u_betty", "u_chen", "u_dan")
    }
    result = await _forget(world, "u_betty", _message("m-1"))
    await world.project()
    assert SECRET not in await _search_text(world, "u_chen")

    undone = await world.memory.undo_group_forget(
        app_id=APP,
        project_id=PROJECT,
        group_id="trip",
        asker_id="u_betty",
        forget_id=result.forget_id,
    )
    await world.project()

    assert undone.applied
    assert set(undone.episodes) == set(result.episodes)
    assert set(undone.withdrawn) == set(result.new_episodes)
    after = _markdown(world)
    for path, text in episodes_before.items():
        assert _entries(after[path]) == _entries(text), path
    profile_before = {p: t for p, t in before.items() if "/profile/group_profile" in p}
    for path, text in profile_before.items():
        assert set(_entries(text).items()) <= set(_entries(after[path]).items())
    assert SECRET in await _search_text(world, "u_chen")
    for member_id, held in profiles_before.items():  # no update ran meanwhile
        back = await world.stores.member_profile.read(world.scope("trip"), member_id)
        assert back == held, member_id
    # The message reads again: nothing forgets it any more.
    assert await world.stores.message_forget.forgotten(world.session("s_trip")) == set()
    again = await world.memory.undo_group_forget(
        app_id=APP,
        project_id=PROJECT,
        group_id="trip",
        asker_id="u_betty",
        forget_id=result.forget_id,
    )
    assert not again.applied


def _entries(text: str) -> dict[str, str]:
    """Each entry of a daily log, by id: what an undo must give back exactly."""
    found = re.findall(r"<!-- entry:(\S+) -->\n(.*?)\n<!-- /entry:\1 -->", text, re.S)
    return dict(found)


async def test_another_members_message_is_refused_and_a_manager_may_forget_it(
    chat: _Chat,
) -> None:
    world = chat.world
    with pytest.raises(ForbiddenError):
        await _forget(world, "u_dan", _message("m-2"))  # Chen's
    with pytest.raises(ForbiddenError):
        await _forget(world, "u_dan", _message("m-2", "m-4"))  # one is his own
    with pytest.raises(ForbiddenError):  # a whole session is the managers'
        await _forget(
            world, "u_betty", GroupForgetTarget(kind="session", session_id="s_trip")
        )
    for stranger in ("u_gone", "u_zoe"):
        with pytest.raises(ForbiddenError):
            await _forget(world, stranger, _message("m-1"), dry_run=True)
    with pytest.raises(NotFoundError):
        await _forget(world, "u_chen", _message("m-99"), dry_run=True)

    owner = await _forget(world, "u_chen", _message("m-4"))  # Dan's, by the owner
    admin = await _forget(world, "u_evan", _message("m-3"))  # Betty's, by an admin
    assert owner.messages == (("s_trip", "m-4"),)
    assert admin.messages == (("s_trip", "m-3"),)
    # The second forget rewrote what the first wrote: undo them in order.
    with pytest.raises(ConflictError):
        await world.memory.undo_group_forget(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_chen",
            forget_id=owner.forget_id,
        )
    with pytest.raises(ForbiddenError):  # the same authorisation as the forget
        await world.memory.undo_group_forget(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_dan",
            forget_id=admin.forget_id,
        )


async def test_an_undo_finds_only_its_own_groups_forget(chat: _Chat) -> None:
    """A user named like the group, or the same group id in another space,
    holds forgets the group's undo must not reach."""
    world = chat.world
    elsewhere = {"target": "message", "target_id": "s_trip", "owner_id": "trip"}
    personal = await world.stores.forget_log.record(
        app_id=APP, project_id=PROJECT, owner_type="user", undo={}, **elsewhere
    )
    other_space = await world.stores.forget_log.record(
        app_id="other", project_id=PROJECT, owner_type="group", undo={}, **elsewhere
    )
    for entry in (personal, other_space):
        with pytest.raises(NotFoundError):
            await world.memory.undo_group_forget(
                app_id=APP,
                project_id=PROJECT,
                group_id="trip",
                asker_id="u_chen",
                forget_id=entry.forget_id,
            )


async def test_a_member_who_leaves_stays_visible_to_the_others(chat: _Chat) -> None:
    world = chat.world
    await world.roster("trip", [leave("u_betty", T0 + 10_000)])
    await world.project()

    assert SECRET in await _search_text(world, "u_chen")
    assert await world.stores.message_forget.forgotten(world.session("s_trip")) == set()
    with pytest.raises(ForbiddenError):  # a former member forgets nothing
        await _forget(world, "u_betty", _message("m-1"))


async def test_a_member_forgets_all_their_messages_and_their_profile(
    chat: _Chat,
) -> None:
    world = chat.world
    target = GroupForgetTarget(kind="member", member_id="u_betty")
    with pytest.raises(ForbiddenError):
        await _forget(world, "u_dan", target, dry_run=True)
    dry = await _forget(world, "u_betty", target, dry_run=True)
    real = await _forget(world, "u_betty", target)
    await _flush(world)

    assert _plan_view(real) == _plan_view(dry)
    assert set(real.messages) == {("s_trip", "m-1"), ("s_trip", "m-3")}
    assert real.member_profiles == ("u_betty",)
    assert (
        await world.stores.member_profile.read(world.scope("trip"), "u_betty") is None
    )
    for asker in ("u_chen", "u_dan"):
        blob = await _search_text(world, asker) + await _gets(world, asker)
        assert SECRET not in blob and "FRIDAYOK" not in blob


async def test_a_session_is_forgotten_by_a_manager_only(chat: _Chat) -> None:
    world = chat.world
    target = GroupForgetTarget(kind="session", session_id="s_trip")
    with pytest.raises(ForbiddenError):
        await _forget(world, "u_betty", target, dry_run=True)
    result = await _forget(world, "u_evan", target)
    await world.project()

    assert result.rewritten == 0 and len(result.episodes) == 1  # nothing left
    for asker in ("u_chen", "u_dan"):
        blob = await _search_text(world, asker) + await _gets(world, asker)
        assert not _markers(blob)


async def test_a_record_is_forgotten_and_absent_from_every_read(chat: _Chat) -> None:
    world = chat.world
    (episode,) = await world.stores.episode.read_for_segments(
        world.scope("trip"),
        list(await world.stores.memcell.cells_in_session(world.session("s_trip"))),
    )
    items = await world.stores.group_profile.items(world.scope("trip"))
    decision = next(i for i in items if i.item.kind == "decision")
    target = GroupForgetTarget(
        kind="record", record_ids=(episode.ref.id, decision.ref.entry_id)
    )
    with pytest.raises(ForbiddenError):  # rests on others' messages too
        await _forget(world, "u_betty", target)
    result = await _forget(world, "u_chen", target)
    await world.project()

    assert set(result.episodes) == {episode.ref.id}
    assert result.group_profile_items == (decision.ref.id,)
    for asker in ("u_chen", "u_evan", "u_betty", "u_dan"):
        blob = await _search_text(world, asker) + await _gets(world, asker)
        assert episode.item.episode not in blob
        assert decision.item.statement not in blob


async def test_a_group_scoped_delete_is_an_owners_forget(chat: _Chat) -> None:
    world = chat.world
    with pytest.raises(ForbiddenError):
        await world.memory.delete_in_group(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_betty",
            session_id="s_trip",
            member_id=None,
        )
    deleted, forgotten = await world.memory.delete_in_group(
        app_id=APP,
        project_id=PROJECT,
        group_id="trip",
        asker_id="u_chen",
        session_id="s_trip",
        member_id=None,
    )
    assert deleted.filters == ["group_id", "session_id"]
    assert deleted.count >= 1 and forgotten.forget_id
    await world.project()
    assert not _markers(await _search_text(world, "u_dan"))
    # Soft: the owner's undo brings it back.
    await world.memory.undo_group_forget(
        app_id=APP,
        project_id=PROJECT,
        group_id="trip",
        asker_id="u_chen",
        forget_id=forgotten.forget_id,
    )
    await world.project()
    assert SECRET in await _search_text(world, "u_dan")


async def test_a_message_forgotten_while_buffered_is_never_read(
    tmp_path: Path,
) -> None:
    model = _Model()
    async with GroupWorld.open(tmp_path, llm=FakeLLMClient(handler=model)) as world:
        await world.roster("trip", ROSTER)
        await _add(world, MESSAGES)  # buffered, not cut yet
        result = await _forget(world, "u_betty", _message("m-1"))
        assert result.messages == (("s_trip", "m-1"),)
        assert result.episodes == ()  # nothing was extracted from it yet

        buffered = await world.memory.search(
            SearchRequest(
                app_id=APP,
                project_id=PROJECT,
                query="plans",
                group_id="trip",
                asker_id="u_chen",
                filters={"session_id": "s_trip"},
            )
        )
        shown = json.dumps(buffered.data.model_dump(mode="json"))
        assert SECRET not in shown and "LISBONVENUE" in shown
        await _flush(world)

        assert not [p for p, t in _markdown(world).items() if SECRET in t]
        assert SECRET not in await _index_rows(world)


async def test_a_forget_whose_model_call_fails_changes_nothing(chat: _Chat) -> None:
    world = chat.world
    before = _markdown(world)
    chat.model.down = True
    with pytest.raises(RuntimeError, match="model down"):
        await _forget(world, "u_betty", _message("m-1"))

    assert _markdown(world) == before
    assert await world.stores.message_forget.forgotten(world.session("s_trip")) == set()


async def test_a_forget_failing_part_way_is_rolled_back(chat: _Chat) -> None:
    """The scene's narrative fails after the new episode version was written."""
    world = chat.world
    await _seed_scene(world)
    before = {p: _entries(t) for p, t in _markdown(world).items()}
    chat.model.failing = {"_ReflectOutput"}
    with pytest.raises(RuntimeError, match="model down"):
        await _forget(world, "u_betty", _message("m-1"))

    assert {p: _entries(t) for p, t in _markdown(world).items()} == before
    assert await world.stores.message_forget.forgotten(world.session("s_trip")) == set()


async def _seed_scene(world: GroupWorld) -> tuple[Any, Any, str]:
    """A second segment, and a scene over both segments' episodes."""
    await _add(world, [("m-5", "u_chen", "LISBONVENUE once more", 5_000)])
    await _flush(world)
    scope = world.scope("trip")
    cells = await world.stores.memcell.cells_in_session(world.session("s_trip"))
    first, second = sorted(
        await world.stores.episode.read_for_segments(scope, list(cells)),
        key=lambda s: s.item.timestamp,
    )
    scene_id = "cl_scene_forget"
    await world.stores.cluster.upsert(
        scope,
        "group_scene",
        Cluster(
            id=scene_id,
            centroid=(1.0,) * 8,
            count=2,
            last_ts=second.item.timestamp,
            members=[first.ref.entry_id, second.ref.entry_id],
        ),
        member_type="episode",
    )
    await world.stores.scene.append(
        scope,
        Scene(
            owner_id="trip",
            scene_id=scene_id,
            version=1,
            member_ids=[first.ref.entry_id, second.ref.entry_id],
            subject="Trip",
            summary=f"Scene of {SECRET}",
            narrative=f"The scene: {SECRET} and LISBONVENUE.",
            timestamp=second.item.timestamp,
            source_start=first.item.source_start or first.item.timestamp,
        ),
        supersedes=None,
        only_if=NoSourceCheck(),
    )
    await world.project()
    return first, second, scene_id


async def test_a_scene_holding_a_rewritten_episode_is_narrated_again(
    chat: _Chat,
) -> None:
    world = chat.world
    first, second, scene_id = await _seed_scene(world)
    result = await _forget(world, "u_betty", _message("m-1"))
    await world.project()

    assert len(result.scenes) == 1 and result.rewritten_scenes == 1
    assert not [p for p, t in _markdown(world).items() if SECRET in t]
    listed = await world.memory.get(
        GetRequest(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_chen",
            memory_type="scene",
        )
    )
    (scene,) = listed.data.scenes
    assert SECRET not in scene.episode and "LISBONVENUE" in scene.episode
    cluster = await world.stores.cluster.get(scene_id)
    assert cluster is not None
    assert first.ref.entry_id not in cluster.members
    assert second.ref.entry_id in cluster.members

    await world.memory.undo_group_forget(
        app_id=APP,
        project_id=PROJECT,
        group_id="trip",
        asker_id="u_betty",
        forget_id=result.forget_id,
    )
    await world.project()
    listed = await world.memory.get(
        GetRequest(
            app_id=APP,
            project_id=PROJECT,
            group_id="trip",
            asker_id="u_chen",
            memory_type="scene",
        )
    )
    (scene,) = listed.data.scenes
    assert SECRET in scene.episode
    restored = await world.stores.cluster.get(scene_id)
    assert restored is not None
    assert restored.members == [first.ref.entry_id, second.ref.entry_id]
