"""SubtleMemory's conflict router: clash -> v5, everything else unchanged.

Pins the reference's best-scoring arm (71.75% against 66.56% for the plain
official prompt -- the only one of its sixteen configurations to beat every
single-prompt variant). The mechanism is easy to get subtly wrong in two ways,
so both are asserted here rather than described in a comment:

1. The signal is not a classifier. v5's first stage extracts atomic facts and
   tags each with a role; ``clash`` is the tag for two facts answering one target
   incompatibly with nothing in the text resolving them. Routing on anything else
   -- the question's category, a keyword -- would be reading the label the
   benchmark is asking us to predict.
2. A question with no clash must come out byte-identical to the unrouted run.
   The reference's gain is an 86:7 exchange (contradictory +22.81pp, the
   remainder -0.61pp); a router that perturbs the remainder spends that margin.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[2] / "benchmarks"
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

sm = importlib.import_module("adapters.subtlememory")


def _reply(*roles: str) -> str:
    facts = ", ".join(
        f'{{"fact": "f{i}", "source": "s{i}", "role": "{r}"}}'
        for i, r in enumerate(roles)
    )
    return f'{{"facts": [{facts}]}}'


def test_clash_routes_to_the_two_stage_answer() -> None:
    tpl, extra = sm.answer_route(
        context="ctx",
        question="q",
        qa_meta={},
        call=lambda _p, _m: _reply("direct", "clash"),
    )
    assert tpl == "ANSWER_PROMPT_V5_FACT_ANSWER"
    assert "facts" in extra
    assert "(clash) f1" in extra["facts"]


def test_no_clash_leaves_the_official_prompt_untouched() -> None:
    """The 1200-odd non-conflict questions must not change at all."""
    tpl, extra = sm.answer_route(
        context="ctx",
        question="q",
        qa_meta={},
        call=lambda _p, _m: _reply("direct", "constraint", "anchor", "background"),
    )
    assert tpl == "ANSWER_PROMPT"
    assert extra == {}


@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        "",
        '{"facts": "not a list"}',
        '{"nope": []}',
        '{"facts": [{"fact": "", "role": "clash"}]}',  # empty fact text is dropped
    ],
)
def test_an_unusable_stage1_reply_falls_back_rather_than_routing(reply: str) -> None:
    """A parse failure must not become a routing decision in either direction.

    Silently treating garbage as "clash" would send the whole benchmark through
    the two-stage path on an empty fact list; treating it as a hard error would
    drop the question. Falling back to the official prompt is the only option
    that leaves the arm interpretable.
    """
    tpl, extra = sm.answer_route(
        context="ctx", question="q", qa_meta={}, call=lambda _p, _m: reply
    )
    assert tpl == "ANSWER_PROMPT"
    assert extra == {}


def test_fenced_json_is_accepted() -> None:
    """Models wrap JSON in ``` fences; the reference tolerates it, so must this."""
    fenced = "```json\n" + _reply("clash") + "\n```"
    tpl, _ = sm.answer_route(
        context="c", question="q", qa_meta={}, call=lambda _p, _m: fenced
    )
    assert tpl == "ANSWER_PROMPT_V5_FACT_ANSWER"


def test_stage1_sees_the_context_and_the_fact_cap() -> None:
    """Stage 1 must be handed the retrieved context, not just the question.

    Routing on the question alone cannot detect a conflict -- the conflict lives
    in the memories.
    """
    seen: list[str] = []

    def call(prompt: str, _max: int) -> str:
        seen.append(prompt)
        return _reply("direct")

    sm.answer_route(
        context="MEMORY-XYZ", question="QUESTION-ABC", qa_meta={}, call=call
    )
    assert len(seen) == 1, "stage 1 must run exactly once per question"
    assert "MEMORY-XYZ" in seen[0]
    assert "QUESTION-ABC" in seen[0]
    assert str(sm.V5_MAX_EXTRACTED_FACTS) in seen[0]


def test_unknown_roles_do_not_masquerade_as_clash() -> None:
    """An invalid role becomes ``background``, never the routing trigger."""
    tpl, _ = sm.answer_route(
        context="c",
        question="q",
        qa_meta={},
        call=lambda _p, _m: _reply("CLASHING", "weird"),
    )
    assert tpl == "ANSWER_PROMPT"


def test_the_v5_prompts_are_the_reference_text() -> None:
    """Both stages carry the placeholders the harness formats them with.

    A prompt copied with a renamed field formats to a KeyError at run time, on
    the first question, after the store is already built.
    """
    for token in ("{context}", "{question}", "{max_facts}"):
        assert token in sm.ANSWER_PROMPT_V5_FACT_EXTRACTOR
    for token in ("{facts}", "{question}"):
        assert token in sm.ANSWER_PROMPT_V5_FACT_ANSWER
    # Stage 2 answers from facts only -- handing it the raw context would make it
    # a different arm than the one that scored 71.75%.
    assert "{context}" not in sm.ANSWER_PROMPT_V5_FACT_ANSWER


def test_the_harness_calls_the_hook_when_an_adapter_defines_it() -> None:
    """`run.py` must consult `answer_route`; without that the adapter is dead code."""
    src = (_BENCH / "run.py").read_text(encoding="utf-8")
    assert 'getattr(_ad, "answer_route", None)' in src
    # The extra stage has to go through the harness's own client so it is billed
    # and counted, not through one the adapter opens itself.
    assert "call=_route_call" in src
