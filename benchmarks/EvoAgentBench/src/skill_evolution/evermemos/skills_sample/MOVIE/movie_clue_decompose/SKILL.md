---
name: movie_clue_decompose
description: >
  Break down complex movie constraints into independent searchable dimensions and tackle them one by one
always: true
---

# Movie Clue Decomposition

## When to use
When a movie question contains multiple intertwined constraints (awards, ratings, directors, actors, era, themes) that are too complex to search all at once.

## Technique
Parse the question and extract all constraints: year range, country, awards, ratings, director/actor characteristics, and plot keywords. Separate them into independent search dimensions.

Start by combining only the 2-3 most promising dimensions in your first search. If that fails, rotate to a different combination. Each dimension should be searchable on its own — never dump all constraints into a single query.

When search results yield specific movie names, immediately pivot to verifying the remaining constraints against that candidate.

## Query Templates
- `"[award name]" movie [year range] [rating]`
- `movie [plot keywords] [era] [country]`
- `"[movie name]" [year] [director name] [specific attribute]`

## Worked Examples

### Example
- Question: A film from 1960-2000 from a country with 25-30k serious assault records in 2003/04
- First, inferred the country (South Africa) from crime statistics
- Then searched: `South African movie film director died 60 70 80 90 years old`
- Why it worked: Decomposed the indirect clue (crime stats → country) from the film search, solving each independently

## Anti-pattern
- **Stuffing all constraints into one query**: `movie 2018 PG-13 festival ReFrame award director born 1970` — too many dimensions at once, search engines return noise. Pick 2-3 dimensions max per query.
