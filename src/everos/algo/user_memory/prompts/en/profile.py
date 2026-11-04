"""English prompts for ProfileExtractor — v2 (atomic items grouped by category).

EXPERIMENTAL PROTOTYPE (exp/profile-atomic-restructure). Structural change from v1:
an item is one atomic fact stated in one or two short sentences, and a category is a
GROUP KEY that many items share — not a unique key whose single item absorbs every
fact of the dimension. Growth therefore lands on the item COUNT axis (where dedupe,
caps and compaction govern) instead of the item LENGTH axis (which v1 left ungoverned
and which produced unreadable multi-hundred-character blobs in production).

Four prompts cover the four calls: ``PROFILE_INITIAL_EXTRACTION_PROMPT`` builds a
profile from a conversation, ``PROFILE_UPDATE_PROMPT`` emits operations against a
stored one, ``PROFILE_COMPACT_PROMPT`` rewrites a profile over its total cap, and
``PROFILE_REGROUP_PROMPT`` reorganises one overcrowded group. The first two inject a
``{{target_user}}`` so extraction is scoped to a single speaker in multi-party
conversations; the last two only rewrite stored items.

Assembled layout is uniform and instructions-first: task line → what a portrait is
(``_PORTRAIT``) → how items are written (``_ITEM_RULES``) → call-specific semantics →
output shape (``_ITEM_SHAPE``) → language anchor → the caller's requirements when given
(``【Additional Requirements】``, see :func:`additional_requirements_section`) → DATA LAST. The task line must not
say "below/above" about anything that is not actually there; every data reference
names its section. Review any edit by reading the RENDERED prompt end to end from the
receiver's seat — flow, redundancy, dangling references — not the template source.

What a profile IS, and how the two buckets divide, is defined once in ``_PORTRAIT`` and
spliced into all four (shared so a rule cannot drift between paths — see v1 history).

Placeholders & rendering: all templates use ``{{double-brace}}`` placeholders and are
rendered via :func:`everos.algo.utils.render_prompt`. Single braces are literal, so the
JSON examples are written exactly as the model should read them, and a placeholder the
renderer cannot substitute raises rather than reaching the model.

Inherited measured lessons from v1 that still bind here:

* **A rule repeated in two places may be load-bearing.** Signals are not partitioned by
  section; do not remove a "duplicate" copy without measuring.
* **An example beats the rule it contradicts.** Keep every example consistent with every
  rule, including in evidence formatting.
* **The naming example has to name the actual subject.** ``_ITEM_RULES`` interpolates
  ``{{target_user}}``; a fixed placeholder name regresses trait extraction to ~zero.
* **Added prohibitions dilute the ones already here.** Measure both directions before
  adding rules.
* **Do not hand the model a list of label names.** Bold literals get copied verbatim,
  not translated into the output language.
"""

from __future__ import annotations

# --------------------------------------------------------------------------------------------------
# Shared blocks. Spliced into all four prompts so a rule cannot hold on one path and not another.
# --------------------------------------------------------------------------------------------------

_PORTRAIT = """\
A profile is a **portrait of a person**, not a log of what happened. Nothing enters it because it occurred — only because it is still true the next time you meet this person. A long, busy conversation can correctly yield nothing at all.

**explicit_info** — what this person **stated** about themselves: their role, the constraints they work under, their environment, the preferences they voiced. One clear statement is enough. It must also still hold next time: "I'm tired today", "I'm in a hurry right now", "I'm annoyed at this bug", "I'm on leave next week" were genuinely said and do not belong in a portrait, and there is nowhere to record when they expire. Anything you would have to infer from what they did does not belong here at all: watching an action tells you the action happened, not that this person holds a standing property. A single operation is an event, not a portrait fact.

**implicit_traits** — latent traits of this person, **inferred** from the conversation: how they decide, what they insist on, what they avoid. A trait is a grounded inference — you read a disposition out of signals and name those signals in its basis. A signal is something this person **chose or asserted**, not an operation they carried out (running a test is an action, not a signal); **one clear signal is enough**, and signals need not share a topic — one disposition can show through on unrelated subjects — but the same statement twice is one signal, not two. Actively look for these: a conversation with any substance should not leave this list empty — leave it empty only for purely operational exchanges.

**Two gates admit an item: it must be a fact about THIS PERSON, and only then does the bucket question arise.** A rule that would hold for whoever sat in their seat — a team process, an SOP, how a system or pipeline works — is nobody's portrait fact, however clearly this person stated it; what may enter is its personal side, when there is one (they set this rule for you to follow; they refuse to work any other way). For what passes that gate, the bucket is decided by whether this person said it, never by what the fact is about: anything traceable to a sentence they uttered is explicit_info — what they refuse to touch, what they insist you do — however much it also reveals about them. implicit_traits holds inferences — a disposition they never stated, read out of what they did say and choose; what someone stated needs no inference. Rephrasing a statement as a disposition ("insists on the team's merge process") is not an inference: it is still the thing they said, and it belongs in explicit_info once, not in both buckets.

**Never restate an action as a capability, duty, skill, familiarity, interest or attitude.** Having run the tests once is not being responsible for testing, and it is not caring about test results or an interest in testing either; asking what a skipped test was is a question, not an attitude towards testing; reading a log is not attentiveness. Do not call this person responsible for, involved in, capable of, familiar with, interested in, attentive to or concerned with anything unless they said so themselves. This is about the claim, not the wording, and holds in every language. A pattern you inferred from behaviour goes to implicit_traits, not to explicit_info reworded.

**A question records nothing but the question, and being told something is not knowing it.** Whatever the assistant explained is the assistant's contribution to the conversation; it never becomes this person's knowledge, familiarity or concern, however specifically they asked for it. Wanting to know why a test was skipped, and being given the reason, leaves the portrait exactly as it was.

An item already on file making such a claim is **not** rescued by the evidence attached to it. That evidence records an occasion; the claim asserts a standing property; an occasion cannot establish one. Such an item was never a portrait item and is to be removed, not rewritten and not merged."""

_ITEM_RULES = """\
**One fact, one item.** An item states a single atomic fact about this person, in one or two short sentences — plain, direct and easy to read. Never chain separate facts together with "also", "as well as" or a comma-list: "prefers small pull requests, insists on rebase-before-merge, and reviews are done in the morning" is THREE items, not one. A statement that genuinely forms one fact stays one item — "works mainly in Python with some Rust" is one fact about their language stack — but the moment a second independent fact appears, it gets its own item.

**A category groups items; it never merges them.** A category names a durable dimension of the person — their role, their daily habits, the tools they work with — and several items sharing one category is the normal, expected shape. A category names the dimension of the FACT itself, never the topic of the conversation it surfaced in: a fact about someone's dev environment stated while laying down collaboration rules still files under their environment. Before naming a category, look at the names already in use: when the fact belongs to a dimension already on file, reuse that exact name. Two names for one dimension scatters the portrait; two facts under one name is what categories are for; one name holding facts of several dimensions has stopped being a category. Category names are coarse and few; the items under them are atomic and many.

**Never name the subject.** A profile is read as being about this one person, so state each item directly: "Works mainly in Python", never "{{target_user}} works mainly in Python", "the user works mainly in Python" or "{{target_user_id}} works mainly in Python". This holds in "evidence" too — give what was said and when, not who said it."""

_TARGET_USER = """\
**TARGET USER: {{target_user}}**
The conversation may have multiple speakers; each line is tagged with the speaker's ``user_id``. {{verb}} only for the speaker whose ``user_id`` equals ``{{target_user_id}}``; anything stated by or about another participant belongs to THAT participant. Do not treat assistant suggestions as this person's traits. The label and id above only locate their lines — neither appears in your output."""

_ITEM_SHAPE = """\
An explicit_info item is {"category", "description", "evidence"}; an implicit_traits item is {"trait", "description", "basis"} — no evidence field, its grounding is the basis. Keep each description to one or two short sentences — concise and plain; if a description needs "also" or a semicolon to hold together, it is more than one item. "evidence" and "basis" are JSON strings, never arrays; when either carries two grounds, combine them inside that one string. "evidence" gives when something was said and quotes it, naming no one — e.g. "2024-10-03: '...'" — and carries **at most two dated quotes**; when a third arrives, keep the two most recent. "basis" names the signals themselves — the choices or assertions you are reading the disposition from, each one findable in the conversation. Restating the requirement ("one clear signal", "multiple instances", "repeated choices") is not a basis; if you cannot name the signals, the trait does not belong here at all."""


_ADDITIONAL_REQUIREMENTS_LEAD = (
    "The caller's additional requirements. They apply on top of the rules above; the reply shape in the "
    "【Output】 section stays as it is."
)


def additional_requirements_section(text: str) -> str:
    """Build what fills a prompt's ``{{additional_requirements}}`` slot when the caller has requirements.

    Every prompt here carries the slot right after its closing language rule, ahead of the data sections; with no
    requirements it takes an empty string and the prompt renders exactly as it would without the slot. The text is
    placed verbatim after a fixed lead-in saying it adds to the bundled rules and leaves the output shape as it is;
    as a rendered value it is never scanned for placeholders, so braces in it stay literal.

    Args:
        text: The caller's additional requirements.

    Returns:
        The 【Additional Requirements】 section, led by the blank line that separates it from the language rule.
    """
    return f"\n\n【Additional Requirements】\n{_ADDITIONAL_REQUIREMENTS_LEAD}\n{text}"


# --------------------------------------------------------------------------------------------------
# INIT — build a profile from a conversation.
# --------------------------------------------------------------------------------------------------

PROFILE_INITIAL_EXTRACTION_PROMPT = (
    """
{{language_rule}}

You are a user-profile analyst. From the conversation given in the 【Conversation】 section at the end, build a profile of the target user, following the rules below.

"""
    + _PORTRAIT
    + "\n\n"
    + _ITEM_RULES
    + "\n\n"
    + _TARGET_USER.replace("{{verb}}", "Build a profile")
    + """

【Output】
"""
    + _ITEM_SHAPE
    + """

Output JSON directly, nothing else:
```json
{"explicit_info": [{"category": "...", "description": "...", "evidence": "..."}],
  "implicit_traits": [{"trait": "...", "description": "...", "basis": "..."}]}
```

{{language_rule}}{{additional_requirements}}

【Conversation】
{{conversation_text}}"""
)


# --------------------------------------------------------------------------------------------------
# UPDATE — emit operations against a stored profile.
# --------------------------------------------------------------------------------------------------

PROFILE_UPDATE_PROMPT = (
    """
{{language_rule}}

You are a user-profile updater. A stored profile and new conversation records are given in the 【Stored Profile】 and 【Conversation Records】 sections at the end; decide which operations to apply to the stored profile, following the rules below.

"""
    + _PORTRAIT
    + "\n\n"
    + _ITEM_RULES
    + "\n\n"
    + _TARGET_USER.replace("{{verb}}", "Update the profile")
    + """

【Operations】
- **update**: an existing item's own fact gains a correction, a contradiction, a sharper wording or fresh evidence. Update acts on the SAME fact only — it never widens an item to absorb a different fact, however close the topic. An updated description still states one fact in one or two short sentences.
- **add**: a fact not yet on file. A new fact belonging to a dimension already on file is still an **add** — reuse that category or trait name and the fact becomes a sibling item under it. Adding a second copy of a fact already on file is always wrong; that is an "update" carrying the fresh evidence.
- **delete**: this person explicitly negates it; it has expired ("traveling next week", already past); it is too trivial to be a portrait ("wants pizza today"); a new statement contradicts it outright; **or the stored item asserts a duty, capability, skill, familiarity, interest or attitude this person never claimed** — delete those on sight, whatever evidence they carry, and do not settle for rewriting them.
- **none**: no operation needed — often the right answer. A conversation made only of operations contains nothing about who this person is, however much of it there is. "none" is complete and correct; manufacturing an operation is not.

【Rules】
1. **Index semantics**: every index resolves against the profile snapshot exactly as numbered in the 【Stored Profile】 section. Operations within one response never shift each other's indices — do not adjust an index to compensate for another operation in the same list. "add" takes no index. Indices for explicit_info and implicit_traits are independent.
2. **Keep an item internally consistent**: an "update" restates the whole item — its category or trait, its description and its evidence or basis, the unchanged ones included — so a description always travels with its grounding and the item never asserts one thing while its grounding supports another.
3. **Add versus update is decided by the FACT, the category inventory only names the group.** Before "add", scan the stored items for the same fact under any wording — found means "update" that item. Not found means "add", filed under the existing category name that covers its dimension (coin a new name only for a genuinely new dimension).

【Output】
"""
    + _ITEM_SHAPE
    + """

Nothing to do:
```json
{
  "operations": [
    {"action": "none"}
  ],
  "update_note": "conversation contains nothing about this person"
}
```
One or more operations:
```json
{
  "operations": [
    {"action": "add", "type": "explicit_info", "data": {"category": "...", "description": "...", "evidence": "..."}},
    {"action": "add", "type": "implicit_traits", "data": {"trait": "...", "description": "...", "basis": "..."}},
    {"action": "update", "type": "explicit_info", "index": 0, "data": {"category": "...", "description": "...", "evidence": "..."}},
    {"action": "delete", "type": "implicit_traits", "index": 1, "reason": "..."}
  ],
  "update_note": "..."
}
```
Every operation object closes with exactly two braces before the comma: one for "data", one for the operation itself.

{{language_rule}}{{additional_requirements}}

【Stored Profile】(each item carries an index)
{{stored_profile}}

【Conversation Records】
{{conversations}}"""
)


# --------------------------------------------------------------------------------------------------
# COMPACT — rewrite a profile that has outgrown its caps.
# --------------------------------------------------------------------------------------------------

PROFILE_COMPACT_PROMPT = (
    """
{{language_rule}}

The stored profile given in the 【Stored Profile】 section at the end is over its caps: it holds {{total_items}} items (explicit_info + implicit_traits combined) against a limit of {{max_items}} TOTAL, no more than {{max_per_category}} items under any one category or trait name, and every description within one or two short sentences. Rewrite it back inside all three, following the rules below.

"""
    + _PORTRAIT
    + "\n\n"
    + _ITEM_RULES
    + """

【How to compact】Work in this order.
1. **Delete everything that was never a portrait item, before touching anything else.** An action restated as a capability, duty, skill, familiarity, interest or attitude ("responsible for testing", "capable of debugging", "involved in version control", "cares about test results", "attentive to CI"); a passing state; anything trivial or already expired. Delete these outright — do NOT merge them: folding twelve "responsible for testing" entries into one still leaves a claim this person never made, and it now looks well-founded. Being numerous is not evidence, and neither is the evidence line attached to each one: it records an occasion, the claim asserts a standing property. Twelve such items become **zero** items, not one.
2. **Move anything misfiled.** A trait that merely restates something this person said is an explicit_info fact — move it to explicit_info under the category covering its dimension, and do not leave a copy behind in implicit_traits.
3. **Split every item that chains several facts.** A description held together by "also", semicolons or a comma-list of independent facts becomes several items — one fact each, one or two short sentences each, all under the same category, each keeping the evidence that belongs to its own fact. Splitting is not a way around the caps: a fact that would then be deleted as trivial should simply be deleted.
4. **Merge only restatements of the SAME fact.** Two items saying one thing in different words become one item keeping the better wording. Distinct facts stay distinct items — never collapse a dimension's items into one summary item; that recreates the blob this rewrite exists to remove.
5. **Regroup the categories.** Near-synonymous names for one dimension all take the name already covering most of its items; a name that has swallowed facts of several dimensions is split into dimension-true categories. Items keep their own content either way — regrouping renames, it never rewrites.
6. **Keep the evidence** of every item you keep — at most the two most recent dated quotes each, naming no one.
7. Prefer deleting a weak item over merging or shortening a strong one. The caps are met by deleting, merging restatements and regrouping — NEVER by inventing a claim or grounding. Ending well under the caps is fine — they are ceilings, not targets.

【Output】
"""
    + _ITEM_SHAPE
    + """

explicit_info + implicit_traits must total no more than {{max_items}} items, with no more than {{max_per_category}} under any one category or trait name. Output JSON directly, nothing else:
```json
{"explicit_info": [{"category": "...", "description": "...", "evidence": "..."}],
  "implicit_traits": [{"trait": "...", "description": "...", "basis": "..."}],
  "compact_note": "what was split, merged and dropped"}
```

{{language_rule}}{{additional_requirements}}

【Stored Profile】
{{profile_text}}"""
)


# --------------------------------------------------------------------------------------------------
# REGROUP — reorganise ONE overcrowded group without touching the rest of the profile.
# --------------------------------------------------------------------------------------------------

PROFILE_REGROUP_PROMPT = (
    """
{{language_rule}}

One group of a stored user profile has grown past its cap: {{count}} items share the {{label_field}} name "{{label}}", against a limit of {{max_per_category}} per name. The group's items are given in the 【Items】 section at the end; reorganise THIS GROUP ONLY, following the rules below — the rest of the profile is not shown and must not be assumed.

"""
    + _PORTRAIT
    + "\n\n"
    + _ITEM_RULES
    + """

【How to regroup】Work in this order.
1. **Merge restatements of the SAME fact** into one item, keeping the better wording and at most the two most recent dated quotes of evidence. Distinct facts stay distinct items.
2. **Split the name if it has swallowed several dimensions.** File each item under the {{label_field}} name of ITS OWN dimension — reuse a name from the 【Other names in use】 section when one fits, otherwise coin a coarse new one. Renaming never rewrites a description.
3. **Delete only what was never a portrait item**: an action restated as a capability, duty, skill, familiarity, interest or attitude; a passing state; anything trivial or expired. Deleting is NOT a way to meet the cap — a real fact stays, filed under a better name.
4. **Never change an item's bucket.** These items stay in the bucket they came from, whatever their wording suggests; never invent content, and keep every claim and grounding within what the shown items support.

【Output】
"""
    + _ITEM_SHAPE
    + """

Return every kept item (renamed or not), nothing else:
```json
{"items": [{"category": "...", "description": "...", "evidence": "..."}],
  "regroup_note": "what was merged, renamed and dropped"}
```

{{language_rule}}{{additional_requirements}}

【Other names in use】(elsewhere in the profile)
{{other_labels}}

【Items】(currently under "{{label}}")
{{items_text}}"""
)
