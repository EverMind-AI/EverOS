---
name: tv_asian_multilingual
description: >
  Search using original-language titles (Korean, Turkish, Hindi) and leverage the "has N alternative titles" pattern common in Asian dramas
always: true
---

# Multilingual Title Search

## When to use
When the question mentions that the drama has multiple titles or alternative names in different languages, or when English-only searches fail for a known Asian drama.

## Technique
Asian dramas frequently have titles in multiple languages: original language, English, and sometimes Spanish, Italian, or German versions. The question clue "has N alternative/other titles" is a strong signal to use this technique.

Search with original-language titles when possible — they are far more discriminative than English translations. For Turkish dramas, try Turkish titles (e.g., "Kış Güneşi" instead of "Winter Sun"). For Korean dramas, try romanized Korean titles.

Also search for the multi-title characteristic itself: `"[show name]" titles different languages also known as`.

## Query Templates
- `"[original language title]" TV series`
- `"[show name]" titles different languages [language list]`
- `[English title] also known as alternative titles drama`

## Worked Examples

### Example
- Question: Non-English pre-2017 series with three other-language titles, character witnesses parents' murder
- Hypothesis: Turkish drama "Winter Sun"
- Verification: `Kış Güneşi Winter Sun titles different languages Spanish Italian German`
- Why it worked: The Turkish title "Kış Güneşi" was far more effective than just "Winter Sun"

## Anti-pattern
- **Searching only in English**: `Winter Sun TV series murder` returns unrelated results. The original Turkish title "Kış Güneşi" immediately identifies the correct show.
