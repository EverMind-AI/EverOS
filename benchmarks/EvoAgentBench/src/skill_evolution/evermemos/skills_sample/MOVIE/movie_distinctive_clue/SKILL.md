---
name: movie_distinctive_clue
description: >
  Prioritize the most discriminative clue in the question — awards, precise rating descriptions, exact quotes — over generic plot summaries
always: true
---

# Movie Distinctive Clue Priority

## When to use
When a movie question contains both generic descriptions (e.g., "about a festival") and highly specific details (e.g., exact rating language, award names, precise quotes). Always lead with the distinctive clue.

## Technique
Rank clues by discriminative power: specific award names > exact rating descriptions/quotes > character traits > plot descriptions > genre/era. The most distinctive clue should anchor your first search.

Precise rating descriptions (e.g., "language and sexual content") are often unique fingerprints that directly identify a movie. Similarly, exact phrases quoted from reviews or official sources are extremely searchable.

Generic descriptions like "about a festival" or "a drama movie" add almost no search value — use them only as secondary filters after finding candidates.

## Query Templates
- `movie "[exact phrase from rating/review]" [year or genre]`
- `"[specific award name]" winner [year range] [one extra constraint]`
- `"[exact quote]" movie film`

## Worked Examples

### Example
- Question: A PG-13 movie about a festival, received ReFrame Stamp, 2018-2023
- Failed approach: `ReFrame Stamp movie festival` — too generic
- Successful query: `movie "about a festival" PG-13 language sexual content`
- Why it worked: The precise rating description "language and sexual content" was far more distinctive than the award name

## Anti-pattern
- **Leading with the weakest clue**: Searching `movie about a festival 2020` when you have a precise rating description available. Always use the most distinctive clue first.
