"""``detect_output_language``: which OutputLanguage a text is written in."""

from __future__ import annotations

import pytest
from everalgo.user_memory import OutputLanguage

from everos.core.utils.language import detect_output_language


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("用户上周去了杭州，参观了西湖并品尝了龙井茶。", "Chinese"),
        ("The user visited Hangzhou last week and tried Longjing tea.", "English"),
        ("Der Nutzer war in Hamburg und hat dort Freunde besucht.", "German"),
        ("ユーザーは先週東京に行き、友人と寿司を食べました。", "Japanese"),
        ("사용자는 지난주 서울에 가서 친구를 만났습니다.", "Korean"),
        ("Пользователь на прошлой неделе ездил в Москву.", "Russian"),
        ("El usuario viajó a Madrid para ver a sus amigos.", "Spanish"),
    ],
)
def test_each_output_language_is_recognised(text: str, expected: str) -> None:
    assert detect_output_language(text) == expected


@pytest.mark.parametrize("text", ["", "   \n", "2026-10-28 12:00", "!!! ... ???"])
def test_text_without_letters_has_no_language(text: str) -> None:
    assert detect_output_language(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Identifiers in Latin do not outvote a Chinese narrative.
        ("用户在 src/everos/service 中修复了 LLMOutputError 的重试逻辑。", "Chinese"),
        # A quoted Chinese name does not relabel an English narrative.
        (
            "The user met 王 at the conference and discussed the roadmap at length.",
            "English",
        ),
        # Kanji outnumber kana, which still settles Japanese.
        ("東京大学の研究室で機械学習を研究している。", "Japanese"),
        # One loanword is not enough evidence for German.
        ("The user ordered a coffee über the counter.", "English"),
    ],
)
def test_mixed_text_follows_its_dominant_language(text: str, expected: str) -> None:
    assert detect_output_language(text) == expected


def test_every_answer_is_an_output_language_value() -> None:
    samples = ["中文", "English", "Deutsch und nicht", "ひらがな", "한국어", "Русский"]
    for sample in samples:
        assert detect_output_language(sample) in set(OutputLanguage)
