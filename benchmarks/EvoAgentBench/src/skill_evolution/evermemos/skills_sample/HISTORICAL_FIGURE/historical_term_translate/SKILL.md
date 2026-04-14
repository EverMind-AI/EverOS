---
name: historical_term_translate
description: >
  Translate descriptive event language in questions into specific historical proper nouns or terms
always: true
---

# Term Translation for Historical Figures

## When to use
When the question describes historical events using indirect language rather than proper nouns (e.g., "exaggerate facts about a region" instead of "tall tales").

## Technique
Questions often paraphrase well-known concepts. Recognize what proper term the description maps to, then search using that term.

Common patterns:
- Behavioral -> cultural terms ("exaggerate facts, tales about rivers" -> "tall tales")
- Life events -> historical events ("immigrated US 1890s, farmer, Washington" -> Japanese American internment)
- Social roles -> specific titles ("healing with diluted substances" -> homeopathy)
- South Asian context clues -> specific regions ("poems deposited in a dry well" -> Kashmiri literary tradition; "multiple master's degrees + political career + marriage 1940s" -> South Asian political leaders like Kashmir/India/Pakistan)

Combine the translated term with time/location constraints. This single step often converts an impossible search into a direct hit.

**Critical: When clues don't match well-known Western historical figures, explicitly try South Asian, Middle Eastern, and African contexts.** Many questions involve figures from these regions who are famous locally but not in English-language sources.

## Query Templates
- `[translated proper term] [time constraint] [location constraint]`
- `"[proper term]" [person characteristics] [era]`
- `[South Asian region] [biographical detail] [era]` (e.g., "Kashmir poetess 18th century")

## Worked Examples

### Example 1: American frontier tall tales
Question: entrepreneur who exaggerated facts about a region, told tales about rivers and camping, 2015 magazine article.
- Translation: "exaggerate facts, tales about rivers and camping" -> "tall tales" (American frontier legends)
- Search: `"tall tales" rivers camping entrepreneur exaggerate region` — direct hit
- Then verified: `"Jim Bridger 2015 magazine article explorer"` — confirmed

### Example 2: Japanese American immigration
Question: immigrated to US 1890-1900, farmer and hotelier in Washington, returned home to marry, 1940s major event.
- Translation: "1890s immigrant, Washington farmer" -> Japanese American history
- Search: `"farmer hotelier Washington state immigrant 1890s"` — found Hirata family

### Example 3: Kashmiri literary tradition
Question: poetess whose poems were deposited in a dry well after death, born 1720-1790, husband was also a poet.
- Translation: "poems deposited in dry well after death" -> Kashmiri literary tradition (preserving manuscripts in wells was a known practice in Kashmir)
- Search: `poetess 1720 1764 manuscripts well hidden after death` → `Arnimal well manuscripts poems dry well` → `Arnimal "dry well" manuscripts deposited after death`
- Why it worked: Recognizing "dry well" + "poetess" + "18th century" as pointing to Kashmir narrowed the search from global to a specific cultural tradition

## Anti-pattern
- **Searching with raw descriptive language**: Using "exaggerate facts about region" verbatim is far less effective than translating to "tall tales". Always attempt term translation before searching.
- **Assuming Western context by default**: When biographical details (multiple degrees, political career, marriage customs) don't quickly match Western figures, try adding "India", "Kashmir", "Pakistan", "Middle East" to the search. South Asian political figures often have multiple advanced degrees and Nobel connections.
