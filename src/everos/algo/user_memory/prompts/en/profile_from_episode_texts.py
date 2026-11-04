"""English prompts for ProfileExtractor's Episode-text path.

Structure: every stage is a pair of messages. The SYSTEM message holds the rules, assembled from blocks in the
order a model works — the stage's role sentence, what the profile holds (``<what_the_profile_holds>``), what never
enters it (``<never_in_the_profile>``, the single home of every exclusion), worked examples of where that line
falls (``<examples>``; INIT and UPDATE only, each in its own reply shape), how this stage does its work
(``<how_to_update>``, ``<how_to_compact>`` or ``<how_to_regroup>``; INIT has none, the blocks before it say
everything INIT needs) and the reply format (``<reply_format>``: one sentence plus a bare, parseable JSON
skeleton, never a fence the model could copy) — and closes with the blocks that vary between requests, least- to
most-variable: the output-language rule (``<output_language>``), the category snapshot
(``<available_categories>``) and, only when the caller supplies them, its additional requirements
(``<additional_requirements>``, see :func:`additional_requirements_block`). Prompt caches match on a prefix: the
language rule takes one of a few fixed texts, the snapshot comes from the caller on every request and the
additional requirements are whatever the caller writes, so this order lets the static prefix be shared across
languages and owners.
The USER message opens with ``<target_person>`` naming the person and then carries only the per-request data,
marked up the same way: narratives as ``<narrative time="...">`` elements, stored items as
``<item index="...">`` elements holding JSON. A rule appears once and never names the person: the rules say "the
target person", so the system text is the same for every owner and every request.

One prompt, one delimiter format. Both messages delimit every block and every input element with an XML-style
tag, carry times and indices as attributes, and keep JSON for data inside a tag; no rule text uses Markdown
emphasis, headings, numbered steps or code fences.
That is what the prompting guides of the current model generation do: the OpenAI GPT-5.x guides wrap each
instruction block in a descriptive snake_case tag, Anthropic's best practices ask for XML tags wherever a prompt
mixes instructions, context and variable input, and Gemini's guidance asks for one format per prompt. With the
input marked up this way it explains itself, so no block describes how to read it.

This path owns its blocks; the MemCell prompts are a separate path with their own definitions.
"""

from __future__ import annotations

from everos.algo.utils import paragraphs, tagged


# Worked examples: their people, jobs and places appear in no test case, so they cannot teach a test's answer.
def _examples(lead: str, *examples: str) -> str:
    """Build an ``<examples>`` block: the lead sentence, then each example in its own ``<example>`` element.

    Args:
        lead: The sentence saying what the examples show and how to use them.
        *examples: Each example's body, already marked up, its ``<reply>`` last.

    Returns:
        The tagged block, a blank line after the lead and between examples.
    """
    return tagged(
        "examples",
        paragraphs(lead, *(tagged("example", example) for example in examples)),
    )


# --------------------------------------------------------------------------------------------------
# Shared blocks
# --------------------------------------------------------------------------------------------------

_TARGET_LINE = "<target_person>{{target_user}}</target_person>"

_WHAT_THE_PROFILE_HOLDS = paragraphs(
    "A profile is what an assistant should know about this person before the next conversation: who they are, "
    "their life and the people in it, their background and skills, what they like and dislike, their long-term "
    "goals, how they want to be treated, and their stable habits and dispositions. Major facts about their past — "
    "where they studied, where they are from, the jobs they have held, the big events in their life — are standing "
    "properties too. It is not a log of what happened.",
    "A profile has two lists: explicit_info and implicit_traits.",
    "explicit_info holds the facts about this person that a narrative states directly. A fact counts as stated "
    "directly only when a narrative explicitly attributes it to this person. Each item has three fields. evidence "
    "is what supports the fact: narrative excerpts or faithful paraphrases, with their dates, in one string, one per "
    "occasion the fact was established or changed; a repeat of the same occasion adds nothing. A paraphrase keeps "
    "the narrative's attribution and invents no wording, date, attribution or identifier. description is the fact "
    "itself: one fact per item, in one or two short sentences without a subject. category is the dimension the fact belongs to (identity, skills, preferences and the like), named "
    "after the fact itself, not after the topic of the narrative it appeared in. Prefer a category listed in "
    "<available_categories>, and when there is no list or none fits, coin a short dimension name.",
    "implicit_traits holds the stable patterns that no single narrative states but that show across several. A "
    "trait is a grounded inference: you read a pattern out of signals and name those signals in its basis. A signal "
    "is a choice or request of theirs that a narrative records, not an operation they carried out; the same thing "
    "retold several times is one signal. Consulting the assistant is what every user does and is not a signal; what "
    "they chose or asked for, and how, can be. A fact a narrative states directly stays in explicit_info; neither "
    "that fact nor the occasions it was stated is a signal, and a trait resting on one is that fact restated. Each "
    "item has three fields. basis names the concrete signals behind the inference — which choices or requests, in "
    'which narratives — in one string; restating the requirement ("several signals", "repeated choices") is not a '
    "basis, and if you cannot name the signals the trait does not belong here. Every signal in a basis is something "
    "a narrative shown records, in its own terms; a basis invents no signal, wording or date to reach a count. "
    "description is what the pattern is: one pattern per item, in one or two short sentences without a subject. "
    "trait is the name of the pattern, a short phrase.",
)

_NEVER_IN_THE_PROFILE_LIST = """\
Whatever the wording and whatever evidence is attached, these never become items:
- A single action, or a single action restated as a capability, skill, duty, interest or attitude. Having run the tests once is not being responsible for testing.
- A passing state, anything trivial, and what this person is working on right now: the task at hand, an ongoing project, a plan with a date.
- A rule that would hold for whoever sat in their seat (a team process, an SOP, how a pipeline works), and a decision taken once for one project.
- A single question, and anything the assistant or another participant said: being told something is not knowing it.
- A fact about anyone other than the target person, even one the target person told, and even inside an item about the relationship. "My brother is an accountant in Shanghai" gives nothing: the accounting and Shanghai are his, and "Has a brother" on its own is an empty item. A relationship enters only as a standing feature of this person's own life — a partner, someone they live with — and then with nothing about the other person.
- A one-off request about format or procedure: how to answer this time, which tool to use for this task."""

# Only the stages that read stored items carry this sentence; INIT has nothing on file to remove.
_STORED_CLAIM_IS_REMOVED = (
    "A stored item that makes such a claim is removed, not rewritten and not merged: its evidence records an "
    "occasion, and an occasion cannot establish a standing property."
)

_LEAVE_UNCERTAIN_OUT = (
    "When attribution, durability or grounding is uncertain, leave the item out."
)

_NEVER_IN_THE_PROFILE_INIT = tagged(
    "never_in_the_profile", paragraphs(_NEVER_IN_THE_PROFILE_LIST, _LEAVE_UNCERTAIN_OUT)
)

_NEVER_IN_THE_PROFILE_STORED = tagged(
    "never_in_the_profile",
    paragraphs(
        _NEVER_IN_THE_PROFILE_LIST, f"{_STORED_CLAIM_IS_REMOVED} {_LEAVE_UNCERTAIN_OUT}"
    ),
)

_JSON_OUTPUT_RULE = "Reply with one JSON object, nothing else."
# The schema fixes which keys an item has, not their order; asking for grounding first is all the prompt can do.
_FIELD_ORDER_RULE = (
    "Write each item's fields in the order shown, evidence or basis first."
)
_REPLY_RULES = f"{_JSON_OUTPUT_RULE} {_FIELD_ORDER_RULE}"

_EXPLICIT_ITEM_SKELETON = '{"evidence": "...", "description": "...", "category": "..."}'
_IMPLICIT_ITEM_SKELETON = '{"basis": "...", "description": "...", "trait": "..."}'
_PROFILE_SKELETON = f'{{"explicit_info": [{_EXPLICIT_ITEM_SKELETON}], "implicit_traits": [{_IMPLICIT_ITEM_SKELETON}]}}'

_OUTPUT_LANGUAGE = tagged("output_language", "{{language_rule}}")

_AVAILABLE_CATEGORIES = tagged("available_categories", "{{available_categories}}")

_ADDITIONAL_REQUIREMENTS_LEAD = (
    "They apply on top of the rules above; the reply shape in <reply_format> stays as it is. Where they conflict "
    "with the rules above, the rules above hold."
)


def _system_prompt(
    role: str,
    never_in_the_profile: str,
    *stage_blocks: str,
    examples: str | None = None,
) -> str:
    """Assemble a stage's system template: the shared static blocks around its own, the per-request blocks last.

    Args:
        role: The stage's role sentence, which opens the message untagged.
        never_in_the_profile: The ``<never_in_the_profile>`` variant for the stage.
        *stage_blocks: The stage's own tagged blocks, in order, ending with ``<reply_format>``.
        examples: The stage's ``<examples>`` block, placed right after ``<never_in_the_profile>``; None for a stage
            that shows none.

    Returns:
        The system template.
    """
    return "\n\n".join(
        (
            role,
            tagged("what_the_profile_holds", _WHAT_THE_PROFILE_HOLDS),
            never_in_the_profile,
            *(() if examples is None else (examples,)),
            *stage_blocks,
            _OUTPUT_LANGUAGE,
            _AVAILABLE_CATEGORIES,
        )
    )


def additional_requirements_block(text: str) -> str:
    """Build the ``<additional_requirements>`` block that closes a system message when the caller has requirements.

    The text is placed verbatim after a fixed lead-in saying it adds to the bundled rules, leaves the output shape
    as it is and yields to the bundled rules where the two conflict. Append the block to an already-rendered
    message: the text is not a template, so braces in it stay literal.

    Args:
        text: The caller's additional requirements.

    Returns:
        The tagged block, the lead-in on its first line.
    """
    return tagged("additional_requirements", f"{_ADDITIONAL_REQUIREMENTS_LEAD}\n{text}")


# --------------------------------------------------------------------------------------------------
# INIT
# --------------------------------------------------------------------------------------------------

_ROLE_INIT = (
    "You are a user-profile analyst. Build the profile of the target person from the Episode narratives. The target "
    "person is named in <target_person>. The Episode narratives are in <episode_narratives>. Follow the rules below."
)

_REPLY_FORMAT_INIT = paragraphs(
    f"Both lists may be empty when the narratives carry nothing about this person. {_REPLY_RULES}",
    _PROFILE_SKELETON,
)

_EXAMPLE_OPERATIONS_ONLY = """\
<episode_narratives>
<narrative time="2026-03-02 09:00 UTC">Ravi asked the assistant to turn a CSV of monthly sales figures into a pivot table and to put the totals in bold. The assistant produced the table; Ravi then asked for the same data as a bar chart.</narrative>
</episode_narratives>
<reply>
{"explicit_info": [], "implicit_traits": []}
</reply>"""

_EXAMPLE_FACTS_AND_A_PATTERN = """\
<episode_narratives>
<narrative time="2026-01-10 10:00 UTC">Ines said she runs a small bookshop in Lisbon and that her cousin, a lawyer, handles the paperwork. She asked how to price a new line of notebooks; when the assistant proposed a price, she asked to see last quarter's sales first.</narrative>
<narrative time="2026-02-14 15:30 UTC">Ines asked for help drafting an email to a book distributor and wanted the figures from her spreadsheet checked before sending it. She mentioned she wants to open a second shop one day. She mentioned she lives with her mother.</narrative>
<narrative time="2026-03-20 11:00 UTC">Ines asked the assistant to rewrite a job advertisement and to keep the tone formal this time.</narrative>
</episode_narratives>
<reply>
{
  "explicit_info": [
    {"evidence": "2026-01-10: said she runs a small bookshop in Lisbon", "description": "Runs a small bookshop in Lisbon.", "category": "Identity"},
    {"evidence": "2026-02-14: said she wants to open a second shop one day", "description": "Wants to open a second shop one day.", "category": "Goals"},
    {"evidence": "2026-02-14: said she lives with her mother", "description": "Lives with her mother.", "category": "Relationships"}
  ],
  "implicit_traits": [
    {"basis": "2026-01-10: asked to see last quarter's sales before accepting a proposed price; 2026-02-14: wanted the spreadsheet figures checked before sending the email", "description": "Checks the numbers before acting on advice.", "trait": "Verifies before deciding"}
  ]
}
</reply>"""

_EXAMPLES_INIT = _examples(
    "Two worked examples. They show what enters the profile and what stays out; copy their judgement, not their "
    "wording.",
    _EXAMPLE_OPERATIONS_ONLY,
    _EXAMPLE_FACTS_AND_A_PATTERN,
)

PROFILE_INITIAL_FROM_EPISODE_TEXTS_SYSTEM_PROMPT = _system_prompt(
    _ROLE_INIT,
    _NEVER_IN_THE_PROFILE_INIT,
    tagged("reply_format", _REPLY_FORMAT_INIT),
    examples=_EXAMPLES_INIT,
)

PROFILE_INITIAL_FROM_EPISODE_TEXTS_USER_PROMPT = "\n\n".join(
    (_TARGET_LINE, tagged("episode_narratives", "{{episode_texts}}"))
)

# --------------------------------------------------------------------------------------------------
# UPDATE
# --------------------------------------------------------------------------------------------------

_ROLE_UPDATE = (
    "You are a user-profile analyst. Update the stored profile of the target person from the Episode narratives. "
    "Express the update as operations on the stored profile. The target person is named in <target_person>. The "
    "stored profile is in <stored_profile>. The Episode narratives are in <episode_narratives>; the stored profile "
    "does not reflect them yet, and they may be older than it. Follow the rules below."
)

_HOW_TO_UPDATE = paragraphs(
    'add or update is decided by the fact, not the wording. Before "add", scan the stored items for the same fact '
    'under any wording — found means "update". Adding a second copy of a stored fact is always wrong.',
    "An update restates the whole item — its category or trait, its description and its evidence or basis, the "
    "unchanged ones included — so a description always travels with its grounding. An update keeps the item on the "
    "same fact: a correction, a sharper wording or a changed value of that fact goes into the item. A fact of another "
    "dimension — a goal beside a job, a habit beside a hobby — is an add, and what <never_in_the_profile> excludes "
    "goes nowhere.",
    "The profile is written in one language, the one <output_language> gives. A stored item written in any other "
    "language is updated: its category or trait, its description and its evidence or basis rewritten in that "
    "language and nothing else changed, even when no narrative mentions it.",
    "delete when this person explicitly negates it, a new statement contradicts it outright, it has expired or is "
    "trivial, or it is on the <never_in_the_profile> list.",
    "none is often the right answer. Narratives made only of operations contain nothing about who this person is, "
    "however many there are; manufacturing an operation is not.",
    "Newer state stands; older narratives only add to it. Every stored item shows observed_at, the time of the "
    "narrative that last established its description, and every narrative shows its own time. A narrative whose "
    'time is before an item\'s observed_at may "add" a fact not yet stored and may "update" that item with '
    'additional evidence, but it must not rewrite the item\'s description or "delete" it. An earlier value of a '
    "dimension that already holds one goes into that item's evidence, never into a new item; the description may "
    "add that history in a short parenthesis after the current value, but the current value stays as written. The "
    "system maintains observed_at and discards operations that break this rule; leave observed_at to the system.",
)

_REPLY_FORMAT_UPDATE = paragraphs(
    f"In update and delete, index is the item's index attribute in <stored_profile>. {_REPLY_RULES}",
    """\
Nothing to do:
{
  "operations": [
    {"action": "none"}
  ]
}""",
    f"""\
One or more operations:
{{
  "operations": [
    {{"action": "add", "type": "explicit_info", "data": {_EXPLICIT_ITEM_SKELETON}}},
    {{"action": "add", "type": "implicit_traits", "data": {_IMPLICIT_ITEM_SKELETON}}},
    {{"action": "update", "type": "explicit_info", "index": 0, "data": {_EXPLICIT_ITEM_SKELETON}}},
    {{"action": "delete", "type": "implicit_traits", "index": 1, "reason": "..."}}
  ]
}}""",
)

_EXAMPLE_NAMED_LANGUAGE_SAME_SUBJECT = """\
With <output_language> naming English:
<stored_profile>
<explicit_info>
<item index="0">{"evidence": "2025-11-03：说自己在京都经营一家花店", "description": "在京都经营一家花店。", "category": "身份", "observed_at": "2025-11-03 10:00 UTC"}</item>
</explicit_info>
<implicit_traits>
</implicit_traits>
</stored_profile>
<episode_narratives>
<narrative time="2026-01-12 14:00 UTC">Hana mentioned that her flower shop in Kyoto is in the Gion district, and asked for a caption for a photo of the new shelf.</narrative>
</episode_narratives>
<reply>
{
  "operations": [
    {"action": "update", "type": "explicit_info", "index": 0, "data": {"evidence": "2025-11-03: said she runs a flower shop in Kyoto; 2026-01-12: said the shop is in the Gion district", "description": "Runs a flower shop in Kyoto's Gion district.", "category": "Identity"}}
  ]
}
</reply>"""

_EXAMPLES_UPDATE = _examples(
    "One worked example. It shows what enters the profile and what stays out; copy its judgement, not its wording.",
    _EXAMPLE_NAMED_LANGUAGE_SAME_SUBJECT,
)

PROFILE_UPDATE_FROM_EPISODE_TEXTS_SYSTEM_PROMPT = _system_prompt(
    _ROLE_UPDATE,
    _NEVER_IN_THE_PROFILE_STORED,
    tagged("how_to_update", _HOW_TO_UPDATE),
    tagged("reply_format", _REPLY_FORMAT_UPDATE),
    examples=_EXAMPLES_UPDATE,
)

PROFILE_UPDATE_FROM_EPISODE_TEXTS_USER_PROMPT = "\n\n".join(
    (
        _TARGET_LINE,
        tagged("stored_profile", "{{stored_profile}}"),
        tagged("episode_narratives", "{{episode_texts}}"),
    )
)

# --------------------------------------------------------------------------------------------------
# COMPACT — the profile as a whole is over its caps.
# --------------------------------------------------------------------------------------------------

_ROLE_COMPACT = (
    "You are a user-profile analyst. Rewrite the stored profile of the target person back inside its caps. The "
    "target person is named in <target_person>. The stored profile is in <stored_profile>. Follow the rules below."
)

# The first requirement carries the one ordering that matters: merging excluded items first turns twelve of them
# into one surviving item instead of none.
_HOW_TO_COMPACT = paragraphs(
    "Delete everything on the <never_in_the_profile> list before merging anything: twelve such items become zero "
    "items, not one.",
    "Move a trait that merely restates a fact a narrative states directly into explicit_info, under the category "
    "covering its dimension.",
    "Split an item that chains several facts into one item each, under the same category, each keeping its own "
    "evidence; a fact that would then be trivial is deleted, not kept.",
    "Merge only restatements of the same fact, keeping the better wording; distinct facts stay distinct items even "
    "within one dimension.",
    "Choose the category of every kept explicit_info item as <what_the_profile_holds> defines it; classification "
    "renames an item and leaves its claim unchanged.",
    "Keep every kept item's grounding in the form <what_the_profile_holds> gives.",
    "Prefer deleting a weak item to merging or shortening a strong one; every claim and grounding you keep comes "
    "from the items you were given, and ending well under the caps is fine.",
)

_REPLY_FORMAT_COMPACT = paragraphs(
    "explicit_info + implicit_traits must total no more than {{max_items}} items, with no more than "
    "{{max_per_category}} under any one category or trait name. "
    f"{_REPLY_RULES}",
    _PROFILE_SKELETON,
)

PROFILE_COMPACT_FROM_EPISODE_TEXTS_SYSTEM_PROMPT = _system_prompt(
    _ROLE_COMPACT,
    _NEVER_IN_THE_PROFILE_STORED,
    tagged("how_to_compact", _HOW_TO_COMPACT),
    tagged("reply_format", _REPLY_FORMAT_COMPACT),
)

PROFILE_COMPACT_FROM_EPISODE_TEXTS_USER_PROMPT = "\n\n".join(
    (
        _TARGET_LINE,
        tagged("stored_profile", "{{profile_text}}", total_items="{{total_items}}"),
    )
)

# --------------------------------------------------------------------------------------------------
# REGROUP — ONE overcrowded group, the rest of the profile unseen.
# --------------------------------------------------------------------------------------------------

_ROLE_REGROUP = (
    "You are a user-profile analyst. Reorganise one group of the stored profile of the target person; the group is "
    "over its cap. The target person is named in <target_person>. The group and its items are in <group>. "
    "Reorganise this group only. The rest of the profile is not shown; work from the items in <group> alone. Follow "
    "the rules below."
)

_HOW_TO_REGROUP = paragraphs(
    "Merge restatements of the same fact into one item, keeping the better wording and the grounding in the form "
    "<what_the_profile_holds> gives.",
    "Split a name that has swallowed several dimensions, choosing each item's category as <what_the_profile_holds> "
    "defines it; renaming leaves every description as it is.",
    "Delete only what is on the <never_in_the_profile> list; every real fact stays, filed under a better name.",
    "Keep every item in the bucket it came from; every claim and grounding comes from the shown items.",
)

REGROUP_REPLY_SKELETONS: dict[str, str] = {
    "explicit_info": f'{{"items": [{_EXPLICIT_ITEM_SKELETON}]}}',
    "implicit_traits": f'{{"items": [{_IMPLICIT_ITEM_SKELETON}]}}',
}
"""The REGROUP reply skeleton for each bucket, filled into the system template's ``{{reply_skeleton}}`` slot.

A group belongs to one bucket and its items come back in that bucket's shape, so the skeleton shows that shape. One
fixed text per bucket keeps the system message identical for every group of a bucket.
"""

_REPLY_FORMAT_REGROUP = paragraphs(
    "The group's cap is {{max_per_category}} items per category or trait name. Return every kept item (renamed or "
    'not) in "items", each with the fields it arrived with (an implicit_traits item keeps basis, description and '
    f"trait). {_REPLY_RULES}",
    "{{reply_skeleton}}",
)

PROFILE_REGROUP_FROM_EPISODE_TEXTS_SYSTEM_PROMPT = _system_prompt(
    _ROLE_REGROUP,
    _NEVER_IN_THE_PROFILE_STORED,
    tagged("how_to_regroup", _HOW_TO_REGROUP),
    tagged("reply_format", _REPLY_FORMAT_REGROUP),
)

PROFILE_REGROUP_FROM_EPISODE_TEXTS_USER_PROMPT = "\n\n".join(
    (
        _TARGET_LINE,
        tagged(
            "group",
            "{{items_text}}",
            bucket="{{bucket}}",
            label_field="{{label_field}}",
            label="{{label}}",
            count="{{count}}",
        ),
    )
)
