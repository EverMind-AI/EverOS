"""Past-time wording: a question that asks about the past by itself.

``[search] past_triggers`` (off by default; docs/configuration.md, "Asking
about the past by its wording"). A search request that does not set
``include_past`` runs as if it did when its question holds past-time
wording. Only the question as asked is read -- never a sub-query the decider
wrote, so whether a search reads the past does not depend on the decider's
wording -- and an ``include_past`` the request sets (true or false) always
wins: the manager does not look for a trigger then.

* **Words.** ``[search] past_trigger_words_en`` / ``_zh``: precise phrases
  ("used to", "back then", and their Chinese counterparts), built in from
  ``config/past_triggers_en.toml`` and ``config/past_triggers_zh.toml``.
  Each entry is a regular expression. An English one matches
  case-insensitively and as whole words (no letter or digit right before or
  after it); a Chinese one anywhere, since Chinese puts no space between
  words. A match that lies inside a match of an exception
  (``past_trigger_exceptions_en`` / ``_zh``) does not count: "is used to",
  a direction such as "walk over", which holds "the past" in Chinese.
* **Explicit past dates** (``[search] past_trigger_dates``): a year, a month
  or a day whose whole period ended before the day the question is asked
  (``asked_at``, else now, in the display timezone): ``in 2019``, ``March
  2024``, ``2024-03-05``, ``the 1990s`` and the Chinese year, month and
  date forms. A year stands alone only after a preposition ("in", "since",
  "during" ...) or as a season of it: a bare number such as "2000 steps" is
  no year. A relative word ("yesterday", "last week") is no date: it asks
  what happened lately, not what something used to be.

The question is read after Unicode NFKC normalisation (full-width letters
and digits become ASCII), with typographic apostrophes made plain and its
whitespace collapsed to single spaces; the trigger is reported as it reads
there. Pure string matching: no model call, nothing stored. The patterns
are compiled once, when the search manager is built.
"""

from __future__ import annotations

__all__ = ["PastTriggers"]

import calendar
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

from everos.core.utils.datetime import from_timestamp_ms

if TYPE_CHECKING:
    from everos.config import GroupSearchSettings

type _Span = tuple[int, int]
type _Day = tuple[int, int, int]
type _Period = Literal["day", "month", "year", "decade", "either_day"]

_FIRST_YEAR: Final = 1900
_LAST_YEAR: Final = 2099
"""Years a date trigger reads; a number outside them is no year."""

_APOSTROPHES: Final = str.maketrans(dict.fromkeys("\u2018\u2019\u02bc", "'"))


@dataclass(frozen=True)
class _Wording:
    """One language's trigger patterns and the exceptions that cancel them."""

    words: tuple[re.Pattern[str], ...]
    exceptions: tuple[re.Pattern[str], ...]

    @classmethod
    def compiled(
        cls, words: Sequence[str], exceptions: Sequence[str], *, english: bool
    ) -> _Wording:
        """English patterns match whole words, case-insensitively; others anywhere."""

        def compile_one(pattern: str) -> re.Pattern[str]:
            if english:
                return re.compile(
                    rf"(?<![a-z0-9])(?:{pattern})(?![a-z0-9])", re.IGNORECASE
                )
            return re.compile(pattern)

        return cls(
            words=tuple(compile_one(p) for p in words),
            exceptions=tuple(compile_one(p) for p in exceptions),
        )

    def find(self, text: str) -> _Span | None:
        """The first match of a word that no match of an exception covers."""
        hits = sorted(_matches(self.words, text), key=lambda s: (s[0], -s[1]))
        if not hits:
            return None
        covers = list(_matches(self.exceptions, text))
        for start, end in hits:
            if not any(lo <= start and end <= hi for lo, hi in covers):
                return start, end
        return None


def _matches(patterns: Iterable[re.Pattern[str]], text: str) -> set[_Span]:
    """Every non-empty match of every pattern, overlapping ones included: one
    per position a pattern matches from."""
    found: set[_Span] = set()
    for pattern in patterns:
        position = 0
        while (match := pattern.search(text, position)) is not None:
            if match.end() > match.start():
                found.add((match.start(), match.end()))
            position = match.start() + 1
    return found


class PastTriggers:
    """The past-time wording that switches a search to the past.

    Built by :meth:`of` from ``[search]``; :meth:`find` reads one question.
    """

    def __init__(self, *, english: _Wording, chinese: _Wording, dates: bool) -> None:
        self._english = english
        self._chinese = chinese
        self._dates = dates

    @classmethod
    def of(cls, settings: GroupSearchSettings) -> PastTriggers | None:
        """The configured triggers; ``None`` with ``[search] past_triggers`` off."""
        if not settings.past_triggers:
            return None
        return cls(
            english=_Wording.compiled(
                settings.past_trigger_words_en,
                settings.past_trigger_exceptions_en,
                english=True,
            ),
            chinese=_Wording.compiled(
                settings.past_trigger_words_zh,
                settings.past_trigger_exceptions_zh,
                english=False,
            ),
            dates=settings.past_trigger_dates,
        )

    def find(self, question: str, *, asked_at_ms: int) -> str | None:
        """The first past-time wording of ``question``, as it reads there.

        Args:
            question: The question as asked.
            asked_at_ms: When it is asked, epoch ms: a date before that day
                is in the past.

        Returns:
            The leftmost trigger -- a word, or a date -- or ``None`` when the
            question holds none.
        """
        text = _normalized(question)
        found = [
            span
            for span in (
                self._english.find(text),
                self._chinese.find(text),
                _past_date(text, _day_of(asked_at_ms)) if self._dates else None,
            )
            if span is not None
        ]
        if not found:
            return None
        start, end = min(found, key=lambda span: (span[0], -span[1]))
        return text[start:end]


def _normalized(question: str) -> str:
    """NFKC, plain apostrophes, single spaces."""
    text = unicodedata.normalize("NFKC", question).translate(_APOSTROPHES)
    return " ".join(text.split())


def _day_of(epoch_ms: int) -> _Day:
    """The day ``epoch_ms`` falls on, in the display timezone."""
    moment = from_timestamp_ms(epoch_ms)
    return moment.year, moment.month, moment.day


# ── Explicit dates ───────────────────────────────────────────────────────────

_MONTH_NAMES: Final = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
_MONTHS: Final = {
    **{name: number for number, name in enumerate(_MONTH_NAMES, start=1)},
    **{name[:3]: number for number, name in enumerate(_MONTH_NAMES, start=1)},
    "sept": 9,
}
"""English month names and their three-letter abbreviations."""

_MONTH_ALTERNATIVES: Final = "|".join(sorted(_MONTHS, key=len, reverse=True))
_MONTH: Final = rf"(?P<month_name>{_MONTH_ALTERNATIVES})\.?"
_YEAR: Final = r"(?P<year>\d{4})"
_MONTH_NUMBER: Final = r"(?P<month>\d{1,2})"
_DAY_NUMBER: Final = r"(?P<day>\d{1,2})"
_ORDINAL: Final = r"(?:st|nd|rd|th)?"

# The Chinese date signs, as escapes so this module stays ASCII: year, month,
# day (written and spoken), and the generation sign that makes a year a decade.
_NIAN: Final = "\u5e74"
_YUE: Final = "\u6708"
_RI: Final = "\u65e5"
_HAO: Final = "\u53f7"
_DAI: Final = "\u4ee3"

_DATE_FORMS: Final[tuple[tuple[re.Pattern[str], _Period], ...]] = tuple(
    (re.compile(pattern, re.IGNORECASE), period)
    for pattern, period in (
        # 2024-03-05, 2024/3/5, 2024.03.05
        (
            rf"(?<!\d){_YEAR}(?P<sep>[-/.]){_MONTH_NUMBER}(?P=sep){_DAY_NUMBER}"
            r"(?!\d)",
            "day",
        ),
        # 05/03/2024: day-month or month-day; past only when both readings are.
        (
            r"(?<!\d)(?P<first>\d{1,2})(?P<sep>[-/.])(?P<second>\d{1,2})(?P=sep)"
            rf"{_YEAR}(?!\d)",
            "either_day",
        ),
        # The Chinese date: year sign, month sign, day sign.
        (
            rf"(?<!\d){_YEAR} ?{_NIAN} ?{_MONTH_NUMBER} ?{_YUE} ?{_DAY_NUMBER} ?"
            rf"[{_RI}{_HAO}]",
            "day",
        ),
        # March 5, 2024
        (rf"(?<![a-z]){_MONTH} {_DAY_NUMBER}{_ORDINAL},? {_YEAR}(?!\d)", "day"),
        # 5 March 2024, the 5th of March, 2024
        (rf"(?<!\d){_DAY_NUMBER}{_ORDINAL} (?:of )?{_MONTH},? {_YEAR}(?!\d)", "day"),
        # 2024-03, 2024/3
        (rf"(?<!\d){_YEAR}[-/.]{_MONTH_NUMBER}(?![-/.]?\d)", "month"),
        # The Chinese month: year sign, month sign.
        (rf"(?<!\d){_YEAR} ?{_NIAN} ?{_MONTH_NUMBER} ?{_YUE}", "month"),
        # March 2024, May of 2019
        (rf"(?<![a-z]){_MONTH},? (?:of )?{_YEAR}(?!\d)", "month"),
        # The Chinese year: year sign, not followed by the generation sign.
        (rf"(?<!\d){_YEAR} ?{_NIAN}(?!{_DAI})", "year"),
        # in 2019, since 2019, during early 2019
        (
            r"(?<![a-z])(?:in|during|since|from|before|until|till|around|by) "
            rf"(?:(?:early|late|mid)[ -])?{_YEAR}(?!\d)",
            "year",
        ),
        # the summer of 2019, the year 2019
        (
            r"(?<![a-z])(?:year|spring|summer|autumn|fall|winter) (?:of )?"
            rf"{_YEAR}(?!\d)",
            "year",
        ),
        # the 1990s ("the 90s" names no century and is not read)
        (r"(?<!\d)(?P<decade>(?:19|20)\d0)'?s(?![a-z])", "decade"),
        # The Chinese decade: year sign and generation sign.
        (rf"(?<!\d)(?P<decade>(?:19|20)\d0) ?{_NIAN}{_DAI}", "decade"),
    )
)
"""Every date form a trigger reads, with the period it names."""


def _past_date(text: str, asked: _Day) -> _Span | None:
    """The first date of ``text`` whose whole period ended before ``asked``."""
    found = [
        (match.start(), match.end())
        for pattern, period in _DATE_FORMS
        for match in pattern.finditer(text)
        if _ended(match, period, asked)
    ]
    return min(found, key=lambda span: (span[0], -span[1])) if found else None


def _ended(match: re.Match[str], period: _Period, asked: _Day) -> bool:
    """Whether the period ``match`` names lies wholly before the day ``asked``."""
    if period == "decade":
        start = int(match["decade"])
        return _FIRST_YEAR <= start <= _LAST_YEAR and start + 9 < asked[0]
    year = int(match["year"])
    if not _FIRST_YEAR <= year <= _LAST_YEAR:
        return False
    if period == "year":
        return year < asked[0]
    if period == "either_day":
        first, second = int(match["first"]), int(match["second"])
        both = ((year, second, first), (year, first, second))
        readings = [day for day in both if _valid(day)]
        return bool(readings) and all(day < asked for day in readings)
    month = (
        _MONTHS[match["month_name"].lower()]
        if match.groupdict().get("month_name")
        else int(match["month"])
    )
    if period == "month":
        return 1 <= month <= 12 and (year, month) < asked[:2]
    day = (year, month, int(match["day"]))
    return _valid(day) and day < asked


def _valid(day: _Day) -> bool:
    """Whether ``day`` is a day of the calendar."""
    year, month, number = day
    return 1 <= month <= 12 and 1 <= number <= calendar.monthrange(year, month)[1]
