"""The one-to-one path, pinned byte for byte (docs/group-memory.md, section 10).

Group memory adds an owner type, an access policy and new request fields; a
request that carries no ``group_id`` must run exactly as it did before them.
This test is that promise made checkable. It drives three one-to-one
conversations through the real assembly -- ``bootstrap``'s memory service over
real markdown, a real system database and a real LanceDB index -- with a
deterministic model and fixed embeddings, then compares what came out with a
golden file produced at the commit *before* group memory:

* every markdown file, byte for byte (the write date and the write clock are
  the only normalised parts: they are the day the test runs);
* every index row the cascade projects from that markdown (vectors as a hash);
* the events the pipeline published;
* search and get responses for the owners, over multi-round retrieval (the
  only search method) with a scripted decider.

The conversations are a LoCoMo excerpt (two human senders in one session: the
per-sender fan-out), the tracked solo assistant chat (one user and an
assistant: the LongMemEval shape) and a one-sender chat with a picture, which
attaches it to the episode call. Since the profile's two-prompt change the
episode call writes no profile item (the profile update reads the pictures
itself, and this test runs the pipeline, not the profile strategy), so the
picture chat writes no ``user.md``; that, and the picture prompt without its
profile block, are what the golden was regenerated for then. It was
regenerated once more for the episode prompt assembled by rule
(``chat.episode``, 2026-11-12): a text cell is still sent everalgo's prompt
byte for byte, so only the picture chat's prompt digest changed, and with it
what is derived from its text -- its fixed embedding -- and nothing else.

Fields added after the golden was recorded are allowed in the responses and
rows only when they are listed in :data:`ADDED_FIELDS` and carry their empty
default, so every addition to a one-to-one output is a reviewed line here.

Regenerate (only on purpose, never to make a failure go away)::

    EVEROS_UPDATE_GOLDEN=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
        tests/unit/test_service/test_one_to_one_golden.py
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
from everalgo.testing import FakeLLMClient
from PIL import Image

from everos.component.tokenizer import build_tokenizer
from everos.config import Settings
from everos.core.utils.datetime import from_iso_format, get_utc_now, to_timestamp_ms
from everos.infra.cascade.registry import match_kind
from everos.infra.persistence import MemoryRoot
from everos.infra.persistence.factories import build_indexes, build_stores
from everos.infra.persistence.index import eq
from everos.memory.deps import MemoryDeps
from everos.memory.ports import Event
from everos.service import GetRequest, SearchRequest
from everos.service.bootstrap import _build_memory_service
from tests.fakes import (
    FakeEmbeddingProvider,
    bundled_instructions,
    handler_deps,
    make_providers,
)
from tests.fakes.storage import make_storage
from tests.fakes.stores import UnusedPort

_REPO = Path(__file__).resolve().parents[3]
_GOLDEN = _REPO / "tests" / "fixtures" / "one_to_one_golden" / "golden.json"
_LOCOMO = (
    _REPO / "tests" / "fixtures" / "long_conversation_locomo_caroline_melanie.json"
)
_SOLO = _REPO / "data" / "solo_chat_en.json"
_DIM = 8
_CUT_EVERY = 6
"""The scripted boundary cuts after every sixth message of the window."""

_GROUP_COLUMNS: Mapping[str, object] = {
    # docs/group-memory.md, section 6.3: empty on every one-to-one row.
    "subject_ids": [],
    "origin": None,
    "source_start": None,
    "source_end": None,
}

_GOLDEN_PICTURE = "b649328550c2651e3d4df10af4cba95c0a8dbc39b7eae4aa86bc57881dedfa99"
"""Content hash of the picture chat's one picture (its ``thumbnail_path`` in the
golden names it)."""

_GROUP_ITEM_FIELDS: Mapping[str, object] = {
    # docs/group-memory.md, section 5.3: "One-to-one responses carry the new
    # fields with empty defaults."
    "group_id": None,
    "subject_ids": [],
    "origin": None,
    "owner_type": None,
    "source": None,
    "member_id": None,
}

ADDED_FIELDS: Mapping[str, Mapping[str, object]] = {
    "episode_row": _GROUP_COLUMNS,
    "image_memory_row": {**_GROUP_COLUMNS, "sender_ids": []},
    # Section 6.3: the member a member-in-group profile row is about.
    "user_profile_row": {"subject_id": None},
    "search": {
        **_GROUP_ITEM_FIELDS,
        "sender_ids": [],
        "scope": None,
        # Step 2 (sections 5.3 and 8.4-8.6): group-only additions.
        "audience": None,
        "group_profile": None,
        "constraints": [],
        "coverage": None,
        # The decider rounds that fell back; empty when every round was decided.
        "degraded": [],
        # The answering model's copy of the picture chat's one picture: its
        # [images] model_edge (1024) copy, named after the golden's own hash.
        "model_path": f"figures/.cache/{_GOLDEN_PICTURE}-1024.jpeg",
    },
    "get": {**_GROUP_ITEM_FIELDS, "scenes": [], "images": [], "group_profile": []},
}
"""Fields a later change added to a one-to-one output, with their empty default.

Keyed by the output they appear in (``episode_row``, ``image_memory_row``,
``user_profile_row``, ``search``, ``get``). A field listed here may appear in
the actual output with exactly this value; any other new field fails. A field
whose value follows from what the golden already holds (``model_path``, a path
named after the golden picture's hash) is listed with that value.
"""


# ── the deterministic model ──────────────────────────────────────────────

_LINE = re.compile(r"^\[[^\]]+\]\s+([^:\n]{1,64}):\s+(.+)$", re.MULTILINE)


def _text_of(messages: Sequence[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            parts.append(content)
            continue
        for part in content or []:
            text = part.get("text") if isinstance(part, dict) else None
            if isinstance(text, str):
                parts.append(text)
    return "\n".join(parts)


def _window(prompt: str) -> int:
    found = re.search(r'<messages count="(\d+)">', prompt)
    if found is None:
        raise AssertionError("not a batch boundary request")
    return int(found.group(1))


def _transcript(prompt: str) -> str:
    """The cell's own transcript: the last ``<conversation>`` block.

    The instructions before it carry worked examples with speaker lines of
    their own; only the block the cell was rendered into is the input.
    """
    start = prompt.rfind("<conversation")
    if start < 0:
        return prompt
    end = prompt.find("</conversation>", start)
    return prompt[start : end if end >= 0 else len(prompt)]


def _episode_from(prompt: str) -> dict[str, str]:
    """An episode that depends on the transcript the model was shown.

    The speaker lines are what the extraction renders from the cell, so a
    change to how a one-to-one cell is rendered changes the episode, and the
    golden catches it. The digest covers the whole prompt, instructions
    included: a one-to-one prompt that changes anywhere changes the episode.
    """
    lines = _LINE.findall(_transcript(prompt))
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
    speakers = sorted({name.strip() for name, _ in lines})
    said = " | ".join(text.strip()[:60] for _, text in lines[:3])
    return {
        "title": f"Talk between {', '.join(speakers) or 'nobody'}",
        "summary": f"{len(lines)} lines ({digest})",
        "content": f"{said} [{digest}]",
    }


def _picture_reply(prompt: str) -> str:
    labels = sorted(set(re.findall(r"\[IMAGE: (img-\d+)\]", prompt)))
    episode = _episode_from(prompt)
    refs = " ".join(f"(a photo, {label})" for label in labels)
    return json.dumps(
        {
            "title": episode["title"],
            "summary": episode["summary"],
            "content": f"{episode['content']} {refs}".strip(),
            "images": [
                {"ref": label, "description": f"A tabby cat on a sill ({label})"}
                for label in labels
            ],
        }
    )


def _answer(messages: Any, *, response_format: type, **_kwargs: Any) -> str:
    name = response_format.__name__
    prompt = _text_of(messages)
    if name == "_BatchReply":
        cuts = list(range(_CUT_EVERY, _window(prompt) + 1, _CUT_EVERY))
        return json.dumps(
            {"reasoning": "scripted", "boundaries": cuts, "tail_complete": False}
        )
    if name == "EpisodeReply":
        return json.dumps(_episode_from(prompt))
    if name == "ImageEpisodeReply":
        return _picture_reply(prompt)
    if name == "_DecisionReply":
        # The multi-round decider: keep the first three candidates, stop.
        return json.dumps({"core": [0, 1, 2], "next_queries": []})
    raise AssertionError(f"unexpected model call: {name}")


# ── the conversations ────────────────────────────────────────────────────


def _locomo_batches() -> list[list[dict[str, Any]]]:
    data = json.loads(_LOCOMO.read_text(encoding="utf-8"))
    return [
        [
            {
                "sender_id": m["sender_id"],
                "role": m["role"],
                "timestamp": m["timestamp"],
                "content": m["content"],
            }
            for m in batch["messages"]
        ]
        for batch in data["batches"][:3]
    ]


def _solo_messages() -> list[dict[str, Any]]:
    data = json.loads(_SOLO.read_text(encoding="utf-8"))
    roles = {
        member: detail.get("role", "user")
        for member, detail in data["session_meta"]["user_details"].items()
    }
    out: list[dict[str, Any]] = []
    for m in data["conversation_list"][:24]:
        out.append(
            {
                "sender_id": m["sender"],
                "sender_name": m.get("sender_name"),
                "role": "assistant"
                if roles.get(m["sender"]) == "assistant"
                else "user",
                "timestamp": to_timestamp_ms(from_iso_format(m["create_time"])),
                "content": m["content"],
            }
        )
    return out


def _png() -> str:
    image = Image.new("RGB", (48, 32), (180, 120, 60))
    for x in range(48):
        image.putpixel((x, x % 32), (20, 20, 20))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _picture_messages() -> list[dict[str, Any]]:
    t0 = 1_761_000_000_000
    return [
        {
            "sender_id": "u_pic",
            "sender_name": "Pia",
            "role": "user",
            "timestamp": t0,
            "content": [
                {"type": "text", "text": "Meet my cat, she sits here all day"},
                {"type": "image", "base64": _png(), "ext": "png", "name": "cat.png"},
            ],
        },
        {
            "sender_id": "bot_pic",
            "sender_name": "Ava",
            "role": "assistant",
            "timestamp": t0 + 60_000,
            "content": "What a calm cat! What is her name?",
        },
        {
            "sender_id": "u_pic",
            "sender_name": "Pia",
            "role": "user",
            "timestamp": t0 + 120_000,
            "content": "Her name is Biscuit.",
        },
    ]


# ── normalisation ────────────────────────────────────────────────────────


def _clock_patterns() -> list[tuple[re.Pattern[str], str]]:
    """The run's own dates and times: the only parts allowed to differ."""
    now = get_utc_now()
    days = {now.date()}
    from everos.core.utils.datetime import today_with_timezone

    days.add(today_with_timezone())
    out: list[tuple[re.Pattern[str], str]] = []
    for day in days:
        iso = day.isoformat()
        out.append((re.compile(rf"{iso}[T ][0-9:.]+(?:Z|[+-]\d\d:\d\d)?"), "<NOW>"))
        out.append((re.compile(iso), "<TODAY>"))
        out.append((re.compile(day.strftime("%Y%m%d")), "<TODAY>"))
    return out


def _normalise(value: Any, patterns: Sequence[tuple[re.Pattern[str], str]]) -> Any:
    if isinstance(value, str):
        for pattern, placeholder in patterns:
            value = pattern.sub(placeholder, value)
        return value
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, list):
        return [_normalise(v, patterns) for v in value]
    if isinstance(value, dict):
        return {
            _normalise(k, patterns): _normalise(v, patterns) for k, v in value.items()
        }
    return value


_MEMCELL_ID = re.compile(r"\bmc_[0-9a-f]{8,}\b")


def _stable_ids(output: Any) -> Any:
    """Memcell ids are random; name each by its first appearance instead.

    The traversal is the sorted JSON rendering, so the numbering depends only
    on the content, never on the run.
    """
    text = json.dumps(output, sort_keys=True, ensure_ascii=False)
    names: dict[str, str] = {}

    def rename(found: re.Match[str]) -> str:
        return names.setdefault(found.group(0), f"mc_<{len(names) + 1}>")

    return json.loads(_MEMCELL_ID.sub(rename, text))


def _vector_digest(vector: Sequence[float] | None) -> str | None:
    if vector is None:
        return None
    rounded = ",".join(f"{float(x):.5f}" for x in vector)
    return hashlib.sha256(rounded.encode("ascii")).hexdigest()[:16]


# ── the run ──────────────────────────────────────────────────────────────


class _RecordingBus:
    def __init__(self) -> None:
        self.events: list[Event] = []

    async def publish(self, events: Sequence[Event]) -> None:
        self.events.extend(events)


def _settings() -> Settings:
    base = Settings()
    return base.model_copy(
        update={
            "embedding": base.embedding.model_copy(update={"dim": _DIM}),
            "lancedb": base.lancedb.model_copy(
                update={"read_consistency_seconds": 0.0}
            ),
            "memorize": base.memorize.model_copy(update={"mode": "chat"}),
        }
    )


def _event_view(event: Event) -> dict[str, Any]:
    keep = (
        "memcell_id",
        "episode_entry_id",
        "owner_id",
        "session_id",
        "app_id",
        "project_id",
        "source",
        "language",
        "episode_timestamp_ms",
    )
    view: dict[str, Any] = {"type": type(event).__name__}
    for name in keep:
        if hasattr(event, name):
            view[name] = getattr(event, name)
    return view


def _row_view(row: Any) -> dict[str, Any]:
    data = row.model_dump(mode="json")
    for name in list(data):
        if "vector" in name:
            data[name] = _vector_digest(getattr(row, name))
    return data


def _markdown_files(root: Path) -> list[Path]:
    """Every user-visible markdown file under ``root``, in path order."""
    return sorted(
        p
        for p in root.rglob("*.md")
        if not p.relative_to(root).parts[0].startswith(".")
    )


def _read_markdown(root: Path, paths: Sequence[Path]) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): p.read_text(encoding="utf-8") for p in paths
    }


async def _run(root: Path) -> dict[str, Any]:
    settings = _settings()
    memory_root = MemoryRoot(root)
    memory_root.ensure()
    storage = make_storage(memory_root, settings)
    await storage.create_schema()
    embedding = FakeEmbeddingProvider(dim=_DIM)
    llm = FakeLLMClient(handler=_answer)
    providers = make_providers(llm=llm, embedding=embedding, rerank=None)
    stores = build_stores(memory_root, storage.sqlite)
    deps = MemoryDeps(
        llm=llm,
        embedding=embedding,
        stores=stores,
        indexes=build_indexes(storage.index),
        instructions=bundled_instructions(),
    )
    bus = _RecordingBus()
    tokenizer = build_tokenizer()
    memory = _build_memory_service(
        settings, providers, deps, tokenizer, bus, UnusedPort("index_eraser")
    )
    adds: list[dict[str, Any]] = []

    async def add(session: str, messages: list[dict[str, Any]], **flags: Any) -> None:
        result = await memory.add(
            {"session_id": session, "messages": messages},
            request_id=f"golden-{len(adds)}",
            async_mode=False,
            defer_extraction=flags.get("defer", False),
        )
        adds.append({"session": session, "status": result.status})

    try:
        locomo = _locomo_batches()
        await add("s_locomo", locomo[0])
        await add("s_locomo", locomo[1])
        await memory.flush(session_id="s_locomo", request_id="golden-flush-1")
        # A deferred tail stays in the buffer for the session-filtered search.
        await add("s_locomo", locomo[2][:3], defer=True)
        solo = _solo_messages()
        for start in range(0, len(solo), 8):
            await add("s_solo", solo[start : start + 8])
        await memory.flush(session_id="s_solo", request_id="golden-flush-2")
        await add("s_picture", _picture_messages())
        await memory.flush(session_id="s_picture", request_id="golden-flush-3")

        markdown = _markdown_files(root)
        handler_bundle = handler_deps(
            memory_root=memory_root,
            tokenizer=tokenizer,
            index=storage.repos,
            embedding=embedding,
        )
        for path in markdown:
            rel = path.relative_to(root).as_posix()
            spec = match_kind(rel)
            if spec is not None:
                await spec.handler_factory(handler_bundle).handle_added_or_modified(rel)
        await storage.index.ensure_business_indexes()

        rows: dict[str, list[dict[str, Any]]] = {}
        for table in ("episode", "image_memory", "user_profile"):
            repo = getattr(storage.repos, table)
            found = await repo.find_where(eq("app_id", "default"), limit=10_000)
            rows[table] = sorted((_row_view(r) for r in found), key=lambda r: r["id"])

        searches: dict[str, Any] = {}
        for owner in ("caroline", "melanie", "user_001", "u_pic"):
            for query in ("support group", "travel to Beijing", "cat"):
                response = await memory.search(
                    SearchRequest(user_id=owner, query=query, top_k=5)
                )
                searches[f"{owner}:{query}"] = response.data.model_dump(mode="json")
        buffered = await memory.search(
            SearchRequest(
                user_id="caroline",
                query="support group",
                top_k=3,
                filters={"session_id": "s_locomo"},
            )
        )
        searches["caroline:session_buffer"] = buffered.data.model_dump(mode="json")
        searches["u_pic:profile"] = (
            await memory.search(
                SearchRequest(
                    user_id="u_pic",
                    query="cat",
                    top_k=3,
                    include_profile=True,
                )
            )
        ).data.model_dump(mode="json")

        gets: dict[str, Any] = {}
        for owner in ("caroline", "user_001", "u_pic"):
            for memory_type in ("episode", "profile"):
                response = await memory.get(
                    GetRequest(user_id=owner, memory_type=memory_type, page_size=50)
                )
                gets[f"{owner}:{memory_type}"] = response.data.model_dump(mode="json")
    finally:
        await storage.aclose()

    raw = {
        "adds": adds,
        "markdown": _read_markdown(root, markdown),
        "rows": rows,
        "events": [_event_view(e) for e in bus.events],
        "search": searches,
        "get": gets,
    }
    return _stable_ids(_normalise(raw, _clock_patterns()))


# ── the comparison ───────────────────────────────────────────────────────


def _added_ok(section: str, key: str, value: object) -> bool:
    allowed = ADDED_FIELDS.get(section, {})
    return key in allowed and allowed[key] == value


def _compare(section: str, expected: Any, actual: Any, path: str) -> Iterator[str]:
    """Every difference between the golden and this run, as readable lines."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in expected:
            if key not in actual:
                yield f"{path}.{key}: missing"
                continue
            yield from _compare(section, expected[key], actual[key], f"{path}.{key}")
        for key in actual:
            if key not in expected and not _added_ok(section, key, actual[key]):
                yield f"{path}.{key}: new field {actual[key]!r}"
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            yield f"{path}: {len(expected)} items expected, {len(actual)} found"
            return
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            yield from _compare(section, e, a, f"{path}[{i}]")
        return
    if expected != actual:
        yield f"{path}: expected {expected!r}, found {actual!r}"


def _section_of(top: str, sub: str) -> str:
    if top == "rows":
        return f"{sub}_row"
    return top


async def test_one_to_one_outputs_match_the_golden(tmp_path: Path) -> None:
    actual = await _run(tmp_path / "root")
    if os.environ.get("EVEROS_UPDATE_GOLDEN") == "1":
        _GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        _GOLDEN.write_text(
            json.dumps(actual, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        pytest.skip("golden regenerated")
    expected = json.loads(_GOLDEN.read_text(encoding="utf-8"))

    # Markdown is the truth: byte for byte, no field may be added.
    assert actual["markdown"] == expected["markdown"]
    assert actual["adds"] == expected["adds"]
    assert actual["events"] == expected["events"]
    problems: list[str] = []
    for top in ("rows", "search", "get"):
        for sub, value in expected[top].items():
            problems.extend(
                _compare(
                    _section_of(top, sub), value, actual[top].get(sub), f"{top}.{sub}"
                )
            )
    assert not problems, "\n".join(problems[:40])


def test_the_golden_covers_every_one_to_one_shape() -> None:
    """The golden is only a gate if it holds what the design names."""
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    paths = list(golden["markdown"])
    # Per-sender fan-out: both LoCoMo speakers own a copy of their episodes.
    assert any("/users/caroline/episodes/" in p for p in paths)
    assert any("/users/melanie/episodes/" in p for p in paths)
    # The assistant of the solo chat owns nothing on the user track.
    assert not any("/users/robot_001/" in p for p in paths)
    # The picture route: a picture memory, and no profile item -- the profile
    # update reads the pictures itself (the strategy, not this pipeline).
    assert any("/users/u_pic/images/" in p for p in paths)
    assert not any(p.endswith("/users/u_pic/user.md") for p in paths)
    assert golden["rows"]["image_memory"]
    assert golden["search"]["caroline:session_buffer"]["unprocessed_messages"]
    assert golden["search"]["u_pic:profile"]["profiles"] == []
