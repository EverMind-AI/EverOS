"""Which language a Reflection merge is asked for and stored with.

The owner's profile language wins. Without one, the members vote with their
stored languages (a member stored without one votes with its text's), and the
winner must hold more than 60% of the members; a closer split names nothing,
and the stored language is then read off the merged text.
"""

from __future__ import annotations

import pytest

from everos.memory.models import Episode
from everos.memory.ports import MemoryRef, Scope, Stored
from everos.memory.reflection._language import (
    LanguageChoice,
    merged_language,
    steer_language,
)

_SCOPE = Scope(app_id="a", project_id="p", owner_id="u1", owner_type="user")


def _member(language: str | None, text: str = "plain text") -> Stored[Episode]:
    return Stored(
        ref=MemoryRef(id="u1_ep", entry_id="ep", md_path="e.md"),
        scope=_SCOPE,
        item=Episode(
            owner_id="u1",
            episode=text,
            summary="",
            timestamp=0,
            subject="s",
            parent_id="mc",
            language=language,
        ),
    )


def test_the_profile_language_wins_over_any_vote() -> None:
    members = [_member("English"), _member("English")]
    assert steer_language("Chinese", members) == LanguageChoice("Chinese", "profile")


@pytest.mark.parametrize(
    ("languages", "expected"),
    [
        (["English", "English"], "English"),
        (["Chinese", "Chinese", "English"], "Chinese"),
        (["Chinese", "English"], None),
        (["Chinese", "Chinese", "Chinese", "English", "English"], None),
        ([], None),
    ],
    ids=["2 of 2", "2 of 3", "1 of 2", "3 of 5 sits on the bar", "no members"],
)
def test_the_members_vote_needs_more_than_sixty_percent(
    languages: list[str], expected: str | None
) -> None:
    members = [_member(language) for language in languages]
    assert steer_language(None, members) == LanguageChoice(expected, "vote")


def test_a_member_stored_without_a_language_votes_with_its_text() -> None:
    members = [_member(None, "用户在讨论周末的行程。"), _member("Chinese")]
    assert steer_language(None, members) == LanguageChoice("Chinese", "vote")


def test_an_unreadable_member_abstains_but_still_counts() -> None:
    """One English vote of two members is no majority."""
    members = [_member(None, "2026-05-17 12:00"), _member("English")]
    assert steer_language(None, members).value is None


def test_a_named_steer_is_what_the_merge_is_stored_with() -> None:
    steer = LanguageChoice("Japanese", "vote")
    assert merged_language(steer, "an English merged text") is steer


def test_without_a_steer_the_merged_text_decides() -> None:
    steer = LanguageChoice(None, "vote")
    assert merged_language(steer, "合并后的叙述") == LanguageChoice("Chinese", "text")
