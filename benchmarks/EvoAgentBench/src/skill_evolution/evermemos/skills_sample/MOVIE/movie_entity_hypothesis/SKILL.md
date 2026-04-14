---
name: movie_entity_hypothesis
description: >
  Use world knowledge to hypothesize specific movie/director/actor names from constraints, then search to verify
always: true
---

# Movie Entity Hypothesis

## When to use
When question clues strongly suggest a specific entity (a particular director, actor, historical event, or movie) that you can guess from world knowledge before searching.

## Technique
Instead of searching generically, use your knowledge to form a concrete hypothesis — a specific movie title, director name, or historical event. Then search to verify that hypothesis against all constraints in the question.

This is especially powerful for questions involving historical events, colonialism, political movements, or well-known filmmakers. The hypothesis turns a broad search into a targeted verification.

If the first hypothesis fails, form a new one from remaining clues rather than falling back to generic searches.

## Query Templates
- `[hypothesized director name] born [year] director filmmaker`
- `[historical event name] [year] [country] documentary film`
- `"[hypothesized movie name]" [year] [constraint to verify]`

## Worked Examples

### Example
- Question: A documentary about an independence activist from a country colonized and annexed in the 20th century
- Hypothesis: Korea (annexed by Japan), Hague Secret Emissary Affair
- Search: `Hague Secret Emissary 1907 Korea emissaries names Netherlands`
- Why it worked: World knowledge directly identified the historical context, skipping dozens of generic searches

## Anti-pattern
- **Searching generically when you can hypothesize**: `colonized country 20th century independence activist documentary` wastes rounds. If you recognize the historical context, guess the entity first and verify.
