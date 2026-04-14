---
name: actor_indian_hypothesis
description: >
  Entity hypothesis — guess candidate Indian actors first, then verify against question constraints
always: true
---

# Entity Hypothesis for Indian Actors

## When to use
When the question describes an Indian/Bollywood actor through indirect clues (awards, controversies, family ties, film records) and you can form a candidate guess early.

## Technique
Clues in Indian actor questions typically point to well-known Bollywood figures. Hypothesize 1-3 candidate names based on the clue pattern, then verify each against every constraint.

Once a candidate name appears in search results, immediately switch to targeted verification: search "[actor name] [specific constraint]" for each condition.

## Query Templates
- `[distinctive event description] actor Bollywood Hindi film`
- `[actor name] [specific constraint condition]` (verification)
- `[actor name] [award name] [year]` (award verification)

## Worked Examples

### Example
Question mentions: never married, US university degree (1800s-founded), debut film in Guinness Records, 2006 flight controversy.
- First search: `"actor misbehavior flight controversy 2006"` — narrowed candidates
- Hypothesized Ameesha Patel, then verified: `"Ameesha Patel flight controversy 2006 misbehavior"` — confirmed
- Key: The 2006 flight incident was distinctive enough to lock in a candidate quickly

### Example
Question mentions: born before 1951, youngest of three brothers, first married 1969.
- Search: `"youngest of three" married 1969 actor director born 1940s`
- Hypothesized Naseeruddin Shah, verified: `"Naseeruddin Shah born youngest siblings married 1969"` — confirmed

## Anti-pattern
**Endless broad searching without forming a hypothesis**: If clues strongly suggest a well-known figure, continuing generic searches wastes steps. Form a candidate guess and verify directly.
