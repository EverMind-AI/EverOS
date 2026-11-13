"""Past-time wording: which questions send a search to the past by themselves.

``[search] past_triggers`` (docs/configuration.md, "Asking about the past by
its wording"); :mod:`everos.memory.search.past_triggers`. Pure matching over
the built-in lists of ``config/past_triggers_en.toml`` and
``config/past_triggers_zh.toml``:

* every built-in phrase fires on a question that asks about how things used
  to be, in English (case-insensitive, whole words) and in Chinese (simplified
  and traditional);
* ordinary questions -- with "before", "earlier", "old", "past", "used",
  "first", a deadline, a direction such as "walk over", the past tense, a
  recent span, a relative day -- fire nothing (at least ten per language);
* an explicit date fires when its whole period (a year, a month, a day) ended
  before the day the question is asked, in the display timezone; a future or
  current one, an impossible one, a bare number and a relative word do not;
* the lists are the settings' defaults; a list that is set replaces its
  built-in one, ``[]`` switches it off, and a pattern that is no regular
  expression, or that matches an empty question, is refused at load.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

import everos.core.utils.datetime as dt_module
from everos.config import GroupSearchSettings, load_settings
from everos.core.utils.datetime import from_iso_format, to_timestamp_ms
from everos.memory.search.past_triggers import PastTriggers

ASKED = to_timestamp_ms(from_iso_format("2026-11-13T00:30:00Z"))
"""When every question below is asked, unless a test says otherwise."""

CONFIG = Path(__file__).resolve().parents[4] / "src/everos/config"


def _triggers(**settings: object) -> PastTriggers:
    built = PastTriggers.of(GroupSearchSettings(past_triggers=True, **settings))
    assert built is not None
    return built


def _find(question: str, **settings: object) -> str | None:
    return _triggers(**settings).find(question, asked_at_ms=ASKED)


# ── English ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("question", "trigger"),
    [
        ("Where did I used to live?", "used to"),
        ("Previously, where did I work?", "Previously"),
        ("Who formerly managed the Lisbon office?", "formerly"),
        ("What did I originally plan for the trip?", "originally"),
        ("What was the plan at first?", "at first"),
        ("What did I do back then?", "back then"),
        ("What games did we play back in the day?", "back in the day"),
        ("What did I like in the past?", "in the past"),
        ("Where did I work at that time?", "at that time"),
        ("Where did we live in those days?", "in those days"),
        ("What did we talk about last time?", "last time"),
        ("Do I not eat meat anymore?", "anymore"),
        ("I don't go to that gym any more, right?", "any more"),
        ("Am I no longer vegetarian?", "no longer"),
        ("Where did I live before I moved to Berlin?", "before I moved"),
        ("What was my job before we met?", "before we met"),
        ("What did she drive before she bought the Tesla?", "before she bought"),
        ("What has my address changed from?", "changed from"),
        ("Which bank have I switched from?", "switched from"),
        ("What was the old address?", "the old"),
        ("What was the previous plan?", "the previous"),
        ("Who was the former owner of the flat?", "the former"),
        ("What was the original budget?", "the original"),
        ("What was my previous job?", "my previous"),
        ("Who was our former landlord?", "our former"),
        ("WHERE DID I USED TO WORK?", "USED TO"),
        ("Where did I  used\tto   live?", "used to"),
        ("我used to住在哪里?", "used to"),
    ],
)
def test_english_past_wording_fires(question: str, trigger: str) -> None:
    assert _find(question) == trigger


@pytest.mark.parametrize(
    "question",
    [
        "What should I pack before the trip next week?",
        "How old is my daughter now?",
        "What time does my flight leave tomorrow?",
        "Can you remind me what the doctor said earlier today?",
        "Is the gym open past 10 pm?",
        "What is the blue folder used to store?",
        "What's this app used to do?",
        "Am I getting used to the new schedule?",
        "I'm used to waking up early, right?",
        "Do I have any more meetings today?",
        "What should I do before I leave the office?",
        "Before we proceed, what is on the agenda?",
        "What did I eat yesterday?",
        "What happened last week?",
        "Where did I go in the past week?",
        "How did I sleep in the past three days?",
        "Which one did I like, the former or the latter?",
        "What is the first thing on my list?",
        "Which laptop do I use to edit photos?",
        "What's the mileage on my 2019 Honda?",
        "How many 2000 calorie days did I log?",
        "Remind me about the unused tickets and the reused bags.",
        "What did I note about oldies music and the previousness of it?",
        # What was said before is current memory, not an old version.
        "What hotel did you recommend in our previous conversation?",
        "Remind me of the vegan place from our previous chat.",
        "When did I go to the farmers' market for the last time?",
    ],
)
def test_ordinary_english_questions_fire_nothing(question: str) -> None:
    assert _find(question) is None


# ── Chinese ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("question", "trigger"),
    [
        ("我以前住在哪里?", "以前"),
        ("我之前的工作是什么?", "之前"),
        ("我从前养过猫吗?", "从前"),
        ("先前定的会议时间是几点?", "先前"),
        ("以往我都怎么过生日?", "以往"),
        ("早先的预算是多少?", "早先"),
        ("过去我常去哪家健身房?", "过去"),
        ("过去几年我换了几份工作?", "过去"),
        ("我曾经养过什么宠物?", "曾经"),
        ("我原来的手机号是多少?", "原来"),
        ("我原本打算几点出发?", "原本"),
        ("原先的地址是哪里?", "原先"),
        ("起初我们计划去哪里旅行?", "起初"),
        ("最初的方案是什么?", "最初"),
        ("我当初为什么离职?", "当初"),
        ("当时我为什么选这家公司?", "当时"),
        ("那时候我住在哪个城市?", "那时"),
        ("那会儿我们住在哪?", "那会儿"),
        ("上次我们讨论了什么?", "上次"),
        ("我上回买的咖啡豆是哪家的?", "上回"),
        ("旧的密码提示是什么?", "旧的"),
        ("我前任经理叫什么?", "前任"),
        ("我不再吃辣了吗?", "不再"),
        ("地址改之前是什么?", "改之前"),
        ("手机换之前用的是什么牌子?", "换之前"),
        # Traditional characters.
        ("我原來的電話號碼是多少?", "原來"),
        ("過去我常去哪裡?", "過去"),
        ("我曾經住在台北嗎?", "曾經"),
        ("當時我們在哪裡吃飯?", "當時"),
        ("手機換之前是什麼牌子?", "換之前"),
    ],
)
def test_chinese_past_wording_fires(question: str, trigger: str) -> None:
    assert _find(question) == trigger


@pytest.mark.parametrize(
    "question",
    [
        # A deadline: before tomorrow, before next week, before five o'clock.
        "我明天之前要交什么材料?",
        "下周以前我需要完成哪些任务?",
        "我五点之前要到哪里?",
        # "Go over" as a direction or a verb.
        "我昨天走过去的那家咖啡店叫什么?",
        "我下午要过去找你吗?",
        "我什么时候过去比较好?",
        "把文件发过去了吗?",
        # Grammar and "acceptable".
        "英语的过去式怎么用?",
        "这个理由说得过去吗?",
        # A recent span counted back from now.
        "我过去一周跑了几次步?",
        "我过去这三天睡得怎么样?",
        # Longer words that hold a trigger.
        "原来如此,那我周末有什么安排?",
        "你可以往我的日程里加一个会议吗?",
        "适当时候提醒我喝水。",
        "当时间到了提醒我出门。",
        "这个课程相当初级吗?",
        "他领先前三名多少分?",
        # What was said before is current memory, not an old version.
        "我们之前的对话里你推荐了哪家餐厅?",
        # Nothing past at all.
        "我的女儿今年几岁了?",
        "我最喜欢吃什么水果?",
        "我昨天晚饭吃了什么?",
        "下周我有什么安排?",
    ],
)
def test_ordinary_chinese_questions_fire_nothing(question: str) -> None:
    assert _find(question) is None


# ── Explicit dates ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("question", "trigger"),
    [
        ("Where was I working in 2019?", "in 2019"),
        ("What did I do during early 2025?", "during early 2025"),
        ("Where did I travel in the summer of 2019?", "summer of 2019"),
        ("What did I do on March 5, 2024?", "March 5, 2024"),
        ("What did I do on 5 March 2024?", "5 March 2024"),
        ("What happened on the 5th of Mar. 2024?", "5th of Mar. 2024"),
        ("What happened on 2025-12-24?", "2025-12-24"),
        ("What did I log on 2026/11/12?", "2026/11/12"),
        ("What was planned for October 2026?", "October 2026"),
        ("What did I buy in 2026-10?", "2026-10"),
        ("What music did I like in the 1990s?", "1990s"),
        ("Was 05/11/2026 a busy day?", "05/11/2026"),
        ("我2023年住在哪里?", "2023年"),
        ("2024年3月5日我去了哪里?", "2024年3月5日"),
        ("我2026年10月做了什么?", "2026年10月"),
        ("2026年11月12号我在哪?", "2026年11月12号"),
        ("1990年代我喜欢什么歌?", "1990年代"),
        ("我２０２３年住在哪里?", "2023年"),
    ],
)
def test_an_explicit_past_date_fires(question: str, trigger: str) -> None:
    assert _find(question) == trigger


@pytest.mark.parametrize(
    "question",
    [
        "What are my plans for 2027?",
        "What did I do in 2026?",
        "Is my dentist appointment on 2026-12-01 still on?",
        "What is on my calendar for 2026-11-13?",
        "What is planned for November 2026?",
        "Is it 12/11/2026 already?",
        "What happened on 2023-02-30?",
        "Where did I live in the 2020s?",
        "I walked 2023 steps, is that a lot?",
        "我2027年有什么计划?",
        "帮我看看2026年12月1日的会议安排。",
        "我2026年11月的日程是什么?",
        "2020年代流行什么?",
        "What did I do yesterday?",
        "我前天去了哪里?",
    ],
)
def test_no_past_date_fires_nothing(question: str) -> None:
    assert _find(question) is None


def test_a_date_is_past_against_the_day_the_question_is_asked() -> None:
    triggers = _triggers()
    in_2025 = to_timestamp_ms(from_iso_format("2025-06-01T00:00:00Z"))
    assert triggers.find("Where was I in 2024?", asked_at_ms=in_2025) == "in 2024"
    assert triggers.find("Where was I in 2025?", asked_at_ms=in_2025) is None
    assert triggers.find("What about 2025-05-31?", asked_at_ms=in_2025)
    assert triggers.find("What about 2025-06-01?", asked_at_ms=in_2025) is None


def test_the_day_asked_is_the_display_timezones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """00:30 UTC on 13 November is still 12 November in Los Angeles."""
    assert _find("What did I log on 2026-11-12?") == "2026-11-12"
    monkeypatch.setenv("EVEROS_MEMORY__TIMEZONE", "America/Los_Angeles")
    load_settings.cache_clear()
    dt_module._display_tz.cache_clear()
    assert _find("What did I log on 2026-11-12?") is None
    assert _find("What did I log on 2026-11-11?") == "2026-11-11"


def test_past_trigger_dates_false_reads_no_date() -> None:
    assert _find("Where was I working in 2019?", past_trigger_dates=False) is None
    assert _find("Where did I used to work in 2019?", past_trigger_dates=False) == (
        "used to"
    )


def test_the_leftmost_trigger_is_named() -> None:
    assert _find("In 2019, where did I used to work?") == "In 2019"
    assert _find("我以前在2019年住在哪里?") == "以前"


# ── The lists ─────────────────────────────────────────────────────────────────


def test_the_built_in_lists_are_the_settings_defaults() -> None:
    settings = GroupSearchSettings()
    assert settings.past_triggers is False
    assert settings.past_trigger_dates is True
    for language in ("en", "zh"):
        with (CONFIG / f"past_triggers_{language}.toml").open("rb") as fh:
            data = tomllib.load(fh)
        words = getattr(settings, f"past_trigger_words_{language}")
        exceptions = getattr(settings, f"past_trigger_exceptions_{language}")
        assert (words, exceptions) == (data["words"], data["exceptions"])
        assert words and exceptions, language


def test_off_builds_nothing() -> None:
    assert PastTriggers.of(GroupSearchSettings()) is None


def test_a_list_that_is_set_replaces_its_built_in_one() -> None:
    mine = {"past_trigger_words_en": ["way back"]}
    assert _find("What did I eat way back?", **mine) == "way back"
    assert _find("Where did I used to live?", **mine) is None
    # The other language keeps its built-in list.
    assert _find("我以前住在哪里?", **mine) == "以前"


def test_an_empty_list_switches_a_language_off() -> None:
    assert _find("我以前住在哪里?", past_trigger_words_zh=[]) is None
    assert _find("Where did I used to live?", past_trigger_words_zh=[]) == "used to"


def test_an_exception_cancels_only_what_lies_inside_it() -> None:
    mine = {"past_trigger_exceptions_en": ["used to (?:it|that)"]}
    assert _find("Am I used to it?", **mine) is None
    assert _find("Am I used to it, as I used to be?", **mine) == "used to"


@pytest.mark.parametrize("pattern", ["(unclosed", "x*", "", "(?:a|)", "(?i)old days"])
def test_a_bad_pattern_is_refused_at_load(pattern: str) -> None:
    for key in (
        "past_trigger_words_en",
        "past_trigger_exceptions_en",
        "past_trigger_words_zh",
        "past_trigger_exceptions_zh",
    ):
        with pytest.raises(ValidationError, match=key):
            GroupSearchSettings(**{key: [pattern]})


def test_the_lists_come_from_the_environment_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVEROS_SEARCH__PAST_TRIGGERS", "true")
    monkeypatch.setenv("EVEROS_SEARCH__PAST_TRIGGER_WORDS_EN", '["way back"]')
    search = load_settings().search
    assert search.past_triggers is True
    assert search.past_trigger_words_en == ["way back"]
    assert search.past_trigger_words_zh == GroupSearchSettings().past_trigger_words_zh
