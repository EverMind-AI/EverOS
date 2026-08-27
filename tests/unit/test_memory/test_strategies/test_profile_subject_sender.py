"""Per-sender profile subjects — ``EVEROS_PROFILE_SUBJECT=sender``.

The default (``owner``) is correct whenever an owner is one person, and the
existing suites cover it. These tests pin the group shape: many people share
one owner (that is what keeps a group chat's retrieval in one partition), so
the subject has to come from the speaker rather than the owner, or the
extractor is handed N people's turns under one name and synthesises a
composite of nobody.

What is asserted here is the mechanism, not the LLM: subject discovery, the
sender re-keying that lets ``aextract`` accept a name, the filename/row-id
split, and the fact that turning the switch off changes nothing.
"""

from __future__ import annotations

import importlib
import itertools
import json
from pathlib import Path

import pytest
from everalgo.types import ChatMessage as AlgoChatMessage
from everalgo.types import MemCell as AlgoMemCell
from everalgo.types import Profile as AlgoProfile

from everos.memory.search.recall.profile import _subject_of

# The module, not the package's re-export: ``@offline_strategy`` replaces the
# module-level name with a ``Strategy`` object, so ``from ... import
# extract_user_profile`` hands back the strategy and none of the helpers.
eup = importlib.import_module("everos.memory.strategies.extract_user_profile")


_SEQ = itertools.count(1)


def _msg(
    *, sender_id: str, sender_name: str | None, role: str = "user", text: str = "hi"
) -> AlgoChatMessage:
    return AlgoChatMessage.model_validate(
        {
            "id": f"m{next(_SEQ)}",
            "role": role,
            "content": text,
            "timestamp": 1_700_000_000_000,
            "sender_id": sender_id,
            "sender_name": sender_name,
        }
    )


def _cell(*messages: AlgoChatMessage) -> AlgoMemCell:
    return AlgoMemCell.model_validate(
        {"items": list(messages), "timestamp": 1_700_000_000_000}
    )


# ── subject discovery ────────────────────────────────────────────────────


def test_subjects_prefer_sender_name_over_the_shared_owner() -> None:
    """A group ingest pins every sender_id to the owner; names carry the person."""
    cell = _cell(
        _msg(sender_id="01", sender_name="Weihua Zhang"),
        _msg(sender_id="01", sender_name="Mingzhi Li"),
        _msg(sender_id="01", sender_name="Weihua Zhang"),
    )
    assert eup._subjects_of([cell]) == ["Weihua Zhang", "Mingzhi Li"]


def test_subjects_skip_assistant_turns() -> None:
    """``aextract`` rejects an assistant outright, so it never becomes a subject."""
    cell = _cell(
        _msg(sender_id="01", sender_name="Lan Ye"),
        _msg(sender_id="01", sender_name="Helper", role="assistant"),
    )
    assert eup._subjects_of([cell]) == ["Lan Ye"]


def test_subjects_fall_back_to_sender_id_when_unnamed() -> None:
    cell = _cell(_msg(sender_id="caroline_conv0", sender_name=None))
    assert eup._subjects_of([cell]) == ["caroline_conv0"]


def test_subjects_span_memcells_in_first_seen_order() -> None:
    cells = [
        _cell(_msg(sender_id="01", sender_name="B")),
        _cell(
            _msg(sender_id="01", sender_name="A"), _msg(sender_id="01", sender_name="B")
        ),
    ]
    assert eup._subjects_of(cells) == ["B", "A"]


# ── the re-keying that makes ``aextract`` accept a name ──────────────────


def test_retarget_rekeys_user_turns_and_leaves_the_source_untouched() -> None:
    """``aextract`` validates sender_id against the memcells' own user senders."""
    cell = _cell(
        _msg(sender_id="01", sender_name="Lan Ye"),
        _msg(sender_id="01", sender_name="Bot", role="assistant"),
    )
    out = eup._retarget([cell], "Lan Ye")

    assert [m.sender_id for m in out[0].items] == ["Lan Ye", "01"]
    # Assistant turns keep the owner id -- only a user turn can be a subject.
    assert [m.sender_id for m in cell.items] == ["01", "01"], "source was mutated"


def test_retarget_output_satisfies_the_extractor_validation() -> None:
    """The exact predicate ``ProfileExtractor.aextract`` raises on."""
    from everalgo.user_memory.profile import _user_senders

    cell = _cell(_msg(sender_id="01", sender_name="Lan Ye"))
    assert "Lan Ye" not in _user_senders([cell]), "precondition"
    assert "Lan Ye" in _user_senders(eup._retarget([cell], "Lan Ye"))


# ── filename / row-id split ──────────────────────────────────────────────


def test_slug_is_filename_safe_and_stable() -> None:
    assert eup._subject_slug("Lan Ye") == "Lan_Ye"
    assert eup._subject_slug("Zhang/Wei..") == "Zhang_Wei"
    assert eup._subject_slug("///") == "unnamed"
    assert eup._subject_slug("Lan Ye") == eup._subject_slug("Lan Ye")


def test_subject_filename_sits_under_the_kind_glob() -> None:
    """The cascade globs each kind once, so the file must match ``user*.md``."""
    from pathlib import PurePosixPath

    from everos.infra.persistence.markdown import UserProfileFrontmatter

    glob = UserProfileFrontmatter.path_glob()
    for name in ("user.md", eup._subject_filename("Lan Ye")):
        rel = f"default_app/default_project/users/01/{name}"
        assert PurePosixPath(rel).match(glob), (rel, glob)


def test_row_id_round_trips_through_the_recaller() -> None:
    """The subject rides the PK because a new column locks old stores out."""
    assert _subject_of("01::Lan Ye", "01") == "Lan Ye"
    assert _subject_of("01", "01") == ""
    # A name containing the delimiter still survives: the prefix is stripped, not split.
    assert _subject_of("01::a::b", "01") == "a::b"


# ── the switch is off by default ─────────────────────────────────────────


def test_owner_is_the_default_subject_mode() -> None:
    assert eup.PROFILE_SUBJECT == eup.SUBJECT_OWNER


# ── per-subject watermark + memcell filtering ────────────────────────────


def test_speaks_in_matches_only_the_subject_s_own_user_turns() -> None:
    cell = _cell(
        _msg(sender_id="01", sender_name="Lan Ye"),
        _msg(sender_id="01", sender_name="Bot", role="assistant"),
    )
    assert eup._speaks_in(cell, "Lan Ye")
    assert not eup._speaks_in(cell, "Jing Lv")
    # An assistant turn is never a subject's evidence, even by name.
    assert not eup._speaks_in(cell, "Bot")


def test_filtering_keeps_the_surrounding_turns_of_a_kept_memcell() -> None:
    """Dropping a memcell drops a meeting, not the other people in it."""
    cell = _cell(
        _msg(sender_id="01", sender_name="Lan Ye", text="ops view"),
        _msg(sender_id="01", sender_name="Weihua Zhang", text="director view"),
    )
    assert eup._speaks_in(cell, "Lan Ye")
    kept = eup._retarget([cell], "Lan Ye")
    texts = [str(m.content) for m in kept[0].items]
    assert any("director view" in t for t in texts), "context was stripped"


# ── extraction trace ─────────────────────────────────────────────────────


def test_trace_is_off_unless_the_env_names_a_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Whitespace is not a path: a blank env value must not create a file."""
    monkeypatch.delenv(eup.PROFILE_TRACE_ENV, raising=False)
    assert eup._trace_path() is None
    monkeypatch.setenv(eup.PROFILE_TRACE_ENV, "   ")
    assert eup._trace_path() is None
    target = tmp_path / "p.jsonl"
    monkeypatch.setenv(eup.PROFILE_TRACE_ENV, str(target))
    assert eup._trace_path() == str(target)
    eup._append_trace({"kind": "profile_extract", "owner_id": "01"})
    assert json.loads(target.read_text())["owner_id"] == "01"


def test_trace_write_failure_never_propagates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Losing a trace line must not lose a profile."""
    monkeypatch.setenv(eup.PROFILE_TRACE_ENV, str(tmp_path / "nope" / "p.jsonl"))
    eup._append_trace({"kind": "profile_extract"})  # parent dir absent -> OSError


def test_trace_record_is_json_serialisable_with_algo_objects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Algo profile items are heterogeneous dicts; ``default=str`` covers them."""
    target = tmp_path / "p.jsonl"
    monkeypatch.setenv(eup.PROFILE_TRACE_ENV, str(target))
    eup._append_trace({"kind": "profile_extract", "obj": object(), "n": {1, 2}})
    assert "profile_extract" in json.loads(target.read_text())["kind"]


def test_profile_shape_reports_what_the_compact_threshold_acts_on() -> None:
    """Counts, not text: 45 items triggers compaction, 30 is the cap."""
    assert eup._profile_shape(None) == {"exists": False}
    prof = AlgoProfile.model_validate(
        {
            "owner_id": "01",
            "summary": "abc",
            "timestamp": 1,
            "explicit_info": [{"category": "a", "description": "b"}],
            "implicit_traits": [{"trait": "x"}, {"trait": "y"}],
        }
    )
    assert eup._profile_shape(prof) == {
        "exists": True,
        "explicit_info": 1,
        "implicit_traits": 2,
        "summary_chars": 3,
    }


# ── INIT language retry ──────────────────────────────────────────────────
#
# Measured on EverMemBench topic 01 (gpt-4.1-mini, 36 INIT calls): 8 of 36 (22%)
# produced a Chinese profile from an all-English corpus, and all 31 subjects
# extracted more than once kept their first language with ZERO exceptions. INIT
# fixes the language, so that one call is what has to be retried.


@pytest.mark.parametrize(
    ("source", "produced", "mismatch"),
    [
        # The real failure: English corpus, Chinese profile. Verbatim from the trace.
        (
            "Good morning everyone. Today we launch the Carbon Emission platform.",
            "Mingzhi Li 是技术部门成员，负责技术架构和数据集成相关工作。",
            True,
        ),
        # The same rule in reverse, so a Chinese corpus is not left unguarded.
        (
            "大家早上好，今天我们启动碳排放平台项目。",
            "Mingzhi Li leads the tech team.",
            True,
        ),
        # Compliant: both English.
        ("Good morning everyone.", "Weihua Zhang is leading the launch.", False),
        # Compliant: both Chinese.
        ("大家早上好。", "张伟华负责启动该项目。", False),
        # A few borrowed product names must NOT trip it -- under the 5% floor.
        (
            "We evaluated the 碳核算 module against ISO 14064 for the whole quarter "
            "and agreed the reporting pipeline needs a second pass before launch.",
            "The team evaluated the accounting module against ISO 14064.",
            False,
        ),
        ("", "", False),
    ],
)
def test_language_mismatch_only_fires_on_a_script_switch(
    source: str, produced: str, mismatch: bool
) -> None:
    assert eup._language_mismatch(source, produced) is mismatch


def test_source_language_names_the_input_for_the_directive() -> None:
    zh = _cell(_msg(sender_id="01", sender_name="A", text="大家早上好，今天启动项目。"))
    en = _cell(_msg(sender_id="01", sender_name="A", text="Good morning everyone."))
    assert eup._source_language([zh]) == "Chinese"
    assert eup._source_language([en]) == "English"


def _profile(summary: str) -> AlgoProfile:
    return AlgoProfile.model_validate(
        {
            "owner_id": "01",
            "summary": summary,
            "timestamp": 1,
            "explicit_info": [],
            "implicit_traits": [],
        }
    )


class _Extractor:
    """Records each call so the test can assert what the retry sent."""

    def __init__(self, *outputs: str) -> None:
        self._outputs = list(outputs)
        self.calls: list[dict] = []

    async def aextract(self, memcells, **kwargs):
        self.calls.append(kwargs)
        return _profile(self._outputs[len(self.calls) - 1])


async def test_init_retries_once_with_the_language_pinned() -> None:
    """The retry appends to the bundled prompt; it must not replace it."""
    cell = _cell(_msg(sender_id="01", sender_name="A", text="Good morning everyone."))
    ex = _Extractor("张三是技术负责人。", "A leads the tech team.")

    profile, retried = await eup._aextract_language_checked(
        ex, [cell], sender_id="A", old_profile=None
    )

    assert retried is True
    assert profile.summary == "A leads the tech team."
    assert len(ex.calls) == 2
    assert ex.calls[0].get("prompt") is None, "first call must use the bundled prompt"
    sent = ex.calls[1]["prompt"]
    assert sent.startswith(eup.PROFILE_INITIAL_EXTRACTION_PROMPT), "prompt was replaced"
    assert "English" in sent
    # A retry is a fresh INIT, not an update onto the rejected profile.
    assert ex.calls[1]["old_profile"] is None


async def test_compliant_init_does_not_retry() -> None:
    cell = _cell(_msg(sender_id="01", sender_name="A", text="Good morning everyone."))
    ex = _Extractor("A leads the tech team.")
    profile, retried = await eup._aextract_language_checked(
        ex, [cell], sender_id="A", old_profile=None
    )
    assert retried is False
    assert len(ex.calls) == 1
    assert profile.summary == "A leads the tech team."


async def test_update_is_never_retried() -> None:
    """UPDATE emits ops onto an existing profile and inherits its language."""
    cell = _cell(_msg(sender_id="01", sender_name="A", text="Good morning everyone."))
    ex = _Extractor("张三是技术负责人。")
    _profile_out, retried = await eup._aextract_language_checked(
        ex, [cell], sender_id="A", old_profile=_profile("张三是技术负责人。")
    )
    assert retried is False
    assert len(ex.calls) == 1


async def test_retry_is_kept_even_when_it_also_fails() -> None:
    """A second sample is no worse than the first; breaking a tie needs a third call."""
    cell = _cell(_msg(sender_id="01", sender_name="A", text="Good morning everyone."))
    ex = _Extractor("张三是技术负责人。", "李四也是中文的。")
    profile, retried = await eup._aextract_language_checked(
        ex, [cell], sender_id="A", old_profile=None
    )
    assert retried is True
    assert profile.summary == "李四也是中文的。"


async def test_retry_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(eup, "LANGUAGE_RETRY", False)
    cell = _cell(_msg(sender_id="01", sender_name="A", text="Good morning everyone."))
    ex = _Extractor("张三是技术负责人。")
    _p, retried = await eup._aextract_language_checked(
        ex, [cell], sender_id="A", old_profile=None
    )
    assert retried is False
    assert len(ex.calls) == 1
