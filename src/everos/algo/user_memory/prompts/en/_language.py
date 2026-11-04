"""The output-language rules every extraction prompt in this package splices in.

A prompt receives ``CALLER_CHOSEN_LANGUAGE_RULE`` when the caller named a language, and otherwise the
fallback its own operator picks: the participant rule for the operators reading a raw conversation, and a
rule of its own for each operator whose input is already-extracted memory text — those inherit a language
rather than judge one, and each names the specific input it inherits from. The profile fallbacks inherit the
language most of that input is written in, so a stored profile that is already mixed converges on one
language instead of keeping the mix. Two of the texts below, ``CALLER_CHOSEN_LANGUAGE_RULE`` and
``PARTICIPANT_LANGUAGE_RULE``, were selected by measurement rather than judgement: eight arms over the same
0.3.2 episode prompt body, 1750 calls each across five models and thirty-five conversations, differing only
in this block. The inheriting rules were not measured — they predate the argument and are carried over from
the prompts that already used them, reworded where they named something those prompts no longer contain
and, for the profile fallbacks, to take the majority language.

Where a rule lands depends on the prompt. Most prompts, the MemCell-path profile prompts among them, splice
it in twice, at head and tail; the profile prompts over Episode texts splice it in once, in the
``<output_language>`` block among the closing blocks of the system message. One text serves both kinds, so
none of them may refer to its own position in the prompt — a sentence pointing above or below is wrong at
one of a head-and-tail prompt's two ends.

``CALLER_CHOSEN_LANGUAGE_RULE`` names the language outright and leaves the model nothing to decide. Zero
drift over 1400 samples spanning seven languages, every interference pattern in the corpus, and all five
models — no scenario, no model, and no language deviated. What these runs credit is the naming, not the
wording: naming the language is also what took profile INIT from 18% wrong-language output over four models
to 0%. The text measured then shouted (a bold ``CRITICAL LANGUAGE RULE`` header, "mandatory", "overrides");
the current text keeps the naming, drops the emphasis, and has not been re-measured. It also says the named
language covers anything already stored — a stored profile, a merged narrative — which is what every inheriting
fallback would otherwise keep; it names no element, so it holds for an operator with nothing stored as well.

Eight rules share a second sentence, ``_KEEP_ORIGINAL_FORM``: the caller-chosen rule, the four fallbacks of
the Episode-path profile stages, and the three ``MEMCELL_*`` fallbacks of the MemCell-path profile maintenance
stages. A profile fallback names the input it inherits from in the markup of its own prompt — ``<stored_profile>``
and ``<group>`` on the Episode path, the 【Stored Profile】 and 【Items】 sections on the MemCell path — so the two
paths cannot share one. The sentence is written once and joined onto each rule where it is defined, so
``build_language_rule`` hands every operator a complete rule and appends nothing. Those eight state the rule
plainly; the remaining rules keep their measured or inherited text and its ``CRITICAL LANGUAGE RULE`` header.

``PARTICIPANT_LANGUAGE_RULE`` asks the model to work the language out. That costs roughly one wrong
language in thirteen, and no rewriting closed the gap: five wordings and one structural variant all landed
between 10% and 26%, each fixing some scenarios while breaking others, because the underlying cause is not
the wording. Chinese characters anywhere in the input pull the output towards Chinese — inside a user ID,
inside a pasted paragraph, inside a single term in an otherwise English sentence — while English inside a
Chinese conversation does not pull back. Six mirrored scenario pairs across eight arms showed no exception.
The residual failures therefore skew one way: a conversation that should yield English is the one at risk.

This rule is nonetheless the best of the self-judging variants, and the only one that never drifted on
plain single-language conversations, which is why it is the fallback rather than one of the others.

The measurements behind those two texts are recorded internally; the corpus and runner live outside the
repository because they need real models and five-figure call counts.
"""

from __future__ import annotations

_KEEP_ORIGINAL_FORM = (
    "Names, user IDs, proper nouns and technical terms keep their original form."
)

# `{{language}}` is filled by `user_memory._language.build_language_rule` before the rule is spliced into a
# prompt: it is the rule's own placeholder, not one any prompt template knows about.
CALLER_CHOSEN_LANGUAGE_RULE = (
    "Write every output field in {{language}}, whatever language the input is in, including anything already stored. "
    + _KEEP_ORIGINAL_FORM
)

EXISTING_PROFILE_LANGUAGE_RULE = (
    "Write every output field in the language most of the stored profile in <stored_profile> is written in, even "
    "when the narratives are in another language. " + _KEEP_ORIGINAL_FORM
)
"""Fallback for the profile UPDATE path: inherit the language most of the profile is already written in.

Inheriting is what stops a later conversation in another language from splitting a profile in half, but it
is still a judgement the model has to make, and a wrong one is unrecoverable — every later update inherits
it in turn. A caller passing `output_language` on every update removes the judgement entirely and can also
correct a profile whose language already went wrong.

The majority, not "the language of the profile", because a profile that is already mixed has no single such
language. Together with ``<how_to_update>``'s rule that a stored item in another language is updated into the
one ``<output_language>`` gives, an update moves a mixed profile to one language instead of keeping the mix.
"""

COMPACTED_PROFILE_LANGUAGE_RULE = (
    "Write every output field in the language most of the stored profile in <stored_profile> is written in. "
    + _KEEP_ORIGINAL_FORM
)
"""Fallback for compaction. Same reasoning as the update rule, minus the incoming conversation.

Compaction rewrites every item, so it is the one path that can change a whole profile's language in a single
call — which makes it the natural place to correct one, and the worst place to guess. Taking the majority, as
the update rule does, brings a mixed profile into one language here.
"""

REGROUPED_PROFILE_LANGUAGE_RULE = (
    "Write every output field in the language most of the items in <group> are written in. "
    + _KEEP_ORIGINAL_FORM
)
"""Fallback for regrouping one overcrowded group. The compaction rule's reasoning, scoped to the group.

Regroup sees one group's items and nothing else of the profile, so the rule names what it actually receives:
the items, not the profile, and takes the language most of them are written in.
"""

MEMCELL_EXISTING_PROFILE_LANGUAGE_RULE = (
    "Write every output field in the language most of the stored profile in the 【Stored Profile】 section is "
    "written in, even when the conversation is in another language. "
    + _KEEP_ORIGINAL_FORM
)
"""MemCell path only: the UPDATE fallback, ``EXISTING_PROFILE_LANGUAGE_RULE`` named for the MemCell prompt's sections.

The MemCell UPDATE prompt gives the stored profile under a 【Stored Profile】 heading and reads conversation records,
not narratives; pointing it at ``<stored_profile>`` would name an element it does not contain.
"""

MEMCELL_COMPACTED_PROFILE_LANGUAGE_RULE = (
    "Write every output field in the language most of the stored profile in the 【Stored Profile】 section is "
    "written in. " + _KEEP_ORIGINAL_FORM
)
"""MemCell path only: the COMPACT fallback, ``COMPACTED_PROFILE_LANGUAGE_RULE`` named for 【Stored Profile】."""

MEMCELL_REGROUPED_PROFILE_LANGUAGE_RULE = (
    "Write every output field in the language most of the items in the 【Items】 section are written in. "
    + _KEEP_ORIGINAL_FORM
)
"""MemCell path only: the REGROUP fallback, ``REGROUPED_PROFILE_LANGUAGE_RULE`` named for the 【Items】 section."""

MERGED_EPISODES_LANGUAGE_RULE = (
    "**CRITICAL LANGUAGE RULE**: You MUST write ALL output in the SAME language as the episodes you are "
    "merging. Merging never changes the language — do not translate. This is mandatory."
)
"""Fallback for merging episodes into one narrative.

The episodes were already extracted, so their language was settled upstream and merging inherits it. That
inheritance is only as good as the episodes agree, though: merging episodes written in different languages
asks the model to pick one, and nothing here says which. A caller who cannot guarantee its episodes share a
language should name the output language rather than leave the pick to the model.
"""

EXISTING_NARRATIVE_LANGUAGE_RULE = (
    "**CRITICAL LANGUAGE RULE**: You MUST write ALL output in the SAME language as the existing narrative "
    "you are updating. Updating never changes the language — do not translate, even if the new episodes are "
    "written in a different language. This is mandatory."
)
"""Fallback for updating a merged narrative. Same shape as the profile update rule.

The narrative already has a language and the incoming episodes may not share it, so the rule pins the
narrative's own — which is what stops an update in a second language from splitting it in half, and is
equally unrecoverable when the language it pins is already wrong. Naming a language is the way out.
"""

SOURCE_TEXT_LANGUAGE_RULE = (
    "**CRITICAL LANGUAGE RULE**: You MUST output in the SAME language <episode_text> itself is written in. "
    "ALL output MUST match that language. This is mandatory."
)
"""Fallback for the operators whose input is already-extracted memory text rather than a conversation.

Deliberately one sentence where ``PARTICIPANT_LANGUAGE_RULE`` is three paragraphs. Everything those
paragraphs deal with — pasted documents, participants writing different languages, an identifier
contradicting the conversation — is a property of a live multi-party conversation. That text has
already been through extraction, so it is a single-language narrative and there is nothing to adjudicate:
the language was settled upstream and this layer inherits it.
"""

EPISODE_PROFILE_INIT_LANGUAGE_RULE = (
    "Write every output field in the language of the narratives in <episode_narratives>; this choice becomes "
    "the profile's language. " + _KEEP_ORIGINAL_FORM
)
"""Fallback for Profile INIT from Episode texts: take the narratives' language, which the profile then keeps.

INIT is the one stage with no earlier profile to inherit from, so it names the input it takes the language from
and says the choice sticks; "every output field" covers the category and trait labels as well as the prose.
"""

PARTICIPANT_LANGUAGE_RULE = (
    "**CRITICAL LANGUAGE RULE**: Write ALL output fields in the language the participants use when they "
    "talk to each other. If they talk in Chinese, write in Chinese; if in English, write in English; if in "
    "Japanese, write in Japanese; and so on for any language. This is mandatory.\n\n"
    "To find that language, read only the message contents — the sentences the participants address to "
    "one another: their questions, answers, opinions, decisions and reactions. Text they merely bring "
    "into the conversation is not theirs: a pasted document, a code block, an error message, a quoted "
    "passage, the contents of a link. Ignore all of it when deciding the language, however much of the "
    "conversation it occupies. Such material usually sits in the same message as the participant's own "
    "words, so split within a message rather than discarding the whole message.\n\n"
    "A foreign word or technical term inside a participant's own sentence does not change the judgement — "
    "go by the language the sentence is built in. Person names, user IDs, proper nouns and technical "
    "terms keep their original form in the output regardless of the output language."
)

PROFILE_INIT_LANGUAGE_RULE = (
    PARTICIPANT_LANGUAGE_RULE + "\n\n"
    "This is the call that fixes the profile's language: later update and compaction calls preserve "
    "whatever language you choose here, so every category, trait and description must be written in that "
    "language too — never in a different language from the rest of the profile."
)
"""Fallback for profile INIT: the participant rule, plus what makes this particular call different.

INIT is the only call that chooses a profile's language rather than inheriting it, and the paths that
inherit it bind every category, trait and description explicitly, so the call that sets the precedent has to
bind them too — otherwise the labels are the one part of a profile no rule ever pins to its language.

The appended sentence is carried over from the pre-``output_language`` INIT prompt rather than newly worded;
only its v1 word for a label is replaced by the fields a v2 item actually has. The participant rule it
extends is untouched, so the eight-arm measurement of that rule still describes the judgement this text asks
for. What the extra sentence does to label-language consistency is not itself measured — the corpus judges
the profile's prose, not its labels.
"""
