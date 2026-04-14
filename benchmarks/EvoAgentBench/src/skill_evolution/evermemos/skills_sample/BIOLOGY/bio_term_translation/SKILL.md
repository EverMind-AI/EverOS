---
name: bio_term_translation
description: >
  Translate colloquial biological descriptions into scientific names or technical terminology for effective species/organism search
always: true
---

# Biology Terminology Translation

## When to use
When a question describes biological features colloquially — e.g., "sharp stones for harvest", "burrowing underground to pupate", "a beloved insect delicacy" — instead of using scientific or common names.

## Technique
Biology questions almost never use the actual species name. They describe organisms through behaviors, appearances, and cultural uses. Your first job is to translate these descriptions into candidate species names (common or scientific).

Think about what organism matches ALL the described features. Generate 2-3 candidate species, then verify each against the full set of clues. Once you have a candidate name, use it directly in subsequent searches.

## Query Templates
- `[morphological features] [behavioral features] [geographic distribution] [usage]`
- `"[candidate species name]" [verification keyword from question]`

## Worked Examples

### Example
- Question: At least two narratives about human awareness of the organism's origins, harvested with sharp stones, associated with sacred/medicinal rituals
- Successful query: `"Yerba Mate" two narratives origins human awareness discovery`
- Why it worked: Identified Yerba Mate from "sharp stones for harvest" + "divine/medicinal ritual", then used the name + question clues

## Anti-pattern
Searching directly with colloquial descriptions (e.g., "sharp stones") instead of translating into species names or academic terms — returns irrelevant results.
