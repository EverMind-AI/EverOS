---
name: football_entity_hypothesis
description: >
  Hypothesize candidate football entities (clubs, players) from constraints, then verify with targeted name-based searches.
always: true
---

# Football: Entity Hypothesis & Verification

## When to use
When search results surface a candidate player or club name that might be the answer, or when you can infer a likely entity from the constraints before searching.

## Technique
When a candidate entity appears in search results, immediately construct a verification query using that name combined with other constraints from the question. Do not stop at the first mention — verify systematically.

Steps:
1. From initial search results, extract any player/club names mentioned.
2. Search `"[entity name] Wikipedia"` or `"[entity name] [constraint from question]"` to pull up the full profile.
3. Check each constraint in the question against the profile: birth year, career timeline, club history, match details.
4. If any constraint fails, discard and try the next candidate.

This also works proactively: if the question describes enough detail (nationality + era + league), hypothesize a candidate from knowledge, then search to verify.

## Query Templates
- `"[player name] Wikipedia birth date career clubs"`
- `"[club name] [player name] [specific event from question]"`
- `"first [nationality] player Premier League [other constraints]"`

## Worked Examples
**Abdisalam Ibrahim:** Question described "first player from his birth country to play in the Premier League." Searched `"first player from African country to play English Premier League"` → found Abdisalam Ibrahim → verified with `"Abdisalam Ibrahim Norway brothers birth date"`.

## Anti-pattern
Finding a club name in results but not following up with a name-based search. Always track and verify every entity that appears — never ignore a lead.
