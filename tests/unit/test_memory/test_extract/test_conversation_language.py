"""The language a cell is extracted in comes from the conversation, never the model.

What this pins (``everos.memory.extract.conversation_language``):

- The human messages decide: an English chat yields English, a Chinese chat
  Chinese, whatever the assistant writes and whatever a picture holds.
- A picture reference is not text: ``[IMAGE: <hash>]`` carries hex letters
  that would otherwise read as English.
- A cell with too little text of its own -- a picture with no caption, a
  two-letter "ok" -- is read together with its session's earlier cells,
  newest first, so a picture-only cell of an English chat is English.
- Only a session with no letter anywhere leaves the choice to the model
  (``None``).
"""

from __future__ import annotations

from everalgo.types import ChatMessage, MemCell
from everalgo.user_memory import OutputLanguage

from everos.memory.extract.conversation_language import (
    MIN_LETTERS,
    conversation_language,
    resolve_cell_language,
)
from everos.memory.models import MemcellRecord
from everos.memory.ports import SessionKey
from tests.fakes.stores import InMemoryMemcellStore

_T0 = 1_700_000_000_000
_PICTURE = "[IMAGE: " + "ab12cd34" * 8 + "]"
_SESSION = SessionKey("default", "default", "s1")


def _message(content: str, *, role: str = "user", at: int = _T0) -> ChatMessage:
    return ChatMessage(
        id=f"m_{at}_{role}",
        role=role,
        content=content,
        timestamp=at,
        sender_id="u1" if role == "user" else "bot",
    )


def _cell(*messages: ChatMessage) -> MemCell:
    return MemCell(items=list(messages), timestamp=messages[-1].timestamp)


def test_an_english_chat_is_english() -> None:
    cell = _cell(_message("I took my dog to the beach this morning, look at him!"))

    assert conversation_language([cell]) == OutputLanguage.ENGLISH


def test_a_chinese_chat_is_chinese() -> None:
    cell = _cell(_message("今天早上我带狗去海边了，你看它多开心！"))

    assert conversation_language([cell]) == OutputLanguage.CHINESE


def test_a_picture_reference_is_not_text() -> None:
    """The hash of a picture would otherwise outvote a short Chinese caption."""
    cell = _cell(_message(f"我的猫 {_PICTURE}"))

    assert conversation_language([cell]) == OutputLanguage.CHINESE


def test_the_human_messages_decide_over_the_assistant() -> None:
    cell = _cell(
        _message("Can you help me plan a weekend trip to the mountains with my kids?"),
        _message("当然可以！我们先确定出发日期。", role="assistant", at=_T0 + 1),
    )

    assert conversation_language([cell]) == OutputLanguage.ENGLISH


def test_a_pasted_code_block_does_not_decide() -> None:
    cell = _cell(
        _message(
            "这段代码为什么报错？\n```python\nfor item in items:\n    print(item)\n```"
        )
    )

    assert conversation_language([cell]) == OutputLanguage.CHINESE


def test_one_long_paste_does_not_outvote_the_speaker() -> None:
    """Votes are per message: the language the speaker keeps writing in wins."""
    paste = "这是一份很长的产品说明书，" * 20
    cell = _cell(
        _message("Here is the manual my landlord sent me.", at=_T0),
        _message(paste, at=_T0 + 1),
        _message("Can you tell me what it says about the heating?", at=_T0 + 2),
    )

    assert conversation_language([cell]) == OutputLanguage.ENGLISH


def test_a_short_cell_reads_its_session_s_earlier_cells() -> None:
    picture_only = _cell(_message(_PICTURE, at=_T0 + 60_000))
    earlier = _cell(_message("We finally moved into the new flat near the river."))

    assert conversation_language([picture_only, earlier]) == OutputLanguage.ENGLISH


def test_a_two_letter_reply_in_a_chinese_session_stays_chinese() -> None:
    short = _cell(_message(f"ok {_PICTURE}", at=_T0 + 60_000))
    earlier = _cell(
        _message("我们终于搬进了河边的新公寓，房东人很好，就是暖气有点问题。")
    )

    assert conversation_language([short, earlier]) == OutputLanguage.CHINESE


def test_a_cell_with_enough_text_ignores_the_session() -> None:
    """A speaker who switches language is followed at the next full cell."""
    current = _cell(
        _message("Ich habe heute den ganzen Tag im Garten gearbeitet und bin müde.")
    )
    earlier = _cell(_message("We finally moved into the new flat near the river."))
    assert sum(ch.isalpha() for ch in current.items[0].content) >= MIN_LETTERS

    assert conversation_language([current, earlier]) == OutputLanguage.GERMAN


def test_a_session_without_human_text_falls_back_to_the_assistant() -> None:
    cell = _cell(
        _message(_PICTURE),
        _message(
            "What a lovely garden, did you plant the roses yourself?",
            role="assistant",
            at=_T0 + 1,
        ),
    )

    assert conversation_language([cell]) == OutputLanguage.ENGLISH


def test_no_letter_anywhere_leaves_the_choice_to_the_model() -> None:
    assert conversation_language([_cell(_message(_PICTURE))]) is None


async def _store_with(*cells: tuple[str, MemCell]) -> InMemoryMemcellStore:
    store = InMemoryMemcellStore()
    await store.insert_many(
        _SESSION,
        [
            MemcellRecord(
                memcell_id=mid,
                cell=cell,
                raw_type="chat",
                message_ids=[],
                sender_ids=[],
            )
            for mid, cell in cells
        ],
    )
    return store


async def test_an_english_chat_s_picture_only_cell_resolves_to_english() -> None:
    """The case the bug report measured: a cell that carries only a picture."""
    earlier = _cell(
        _message("I spent the weekend repainting the kitchen cabinets.", at=_T0)
    )
    current = _cell(_message(_PICTURE, at=_T0 + 60_000))
    store = await _store_with(("mc_1", earlier), ("mc_2", current))

    language = await resolve_cell_language(
        current, memcell_id="mc_2", session=_SESSION, memcells=store
    )

    assert language == OutputLanguage.ENGLISH


async def test_a_cell_with_enough_text_reads_no_history() -> None:
    current = _cell(
        _message(
            "I spent the weekend repainting the kitchen cabinets, they look great now."
        )
    )

    class _NoHistory(InMemoryMemcellStore):
        async def recent_in_session(
            self, *args: object, **kwargs: object
        ) -> list[MemCell]:
            raise AssertionError("history must not be read")

    language = await resolve_cell_language(
        current, memcell_id="mc_2", session=_SESSION, memcells=_NoHistory()
    )

    assert language == OutputLanguage.ENGLISH


async def test_history_is_this_session_s_and_no_later_than_the_cell() -> None:
    current = _cell(_message(_PICTURE, at=_T0 + 60_000))
    later = _cell(
        _message("这是之后才发生的对话，不能决定前面图片的语言。", at=_T0 + 120_000)
    )
    other = _cell(_message("这是另一个会话的内容，和这里没有关系。", at=_T0))
    store = await _store_with(
        (
            "mc_1",
            _cell(_message("Our neighbours invited us over for a barbecue.", at=_T0)),
        ),
        ("mc_2", current),
        ("mc_3", later),
    )
    await store.insert_many(
        SessionKey("default", "default", "s2"),
        [
            MemcellRecord(
                memcell_id="mc_x",
                cell=other,
                raw_type="chat",
                message_ids=[],
                sender_ids=[],
            )
        ],
    )

    language = await resolve_cell_language(
        current, memcell_id="mc_2", session=_SESSION, memcells=store
    )

    assert language == OutputLanguage.ENGLISH
