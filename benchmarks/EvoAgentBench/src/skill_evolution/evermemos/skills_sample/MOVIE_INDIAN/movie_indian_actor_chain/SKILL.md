---
name: movie_indian_actor_chain
description: >
  Locate South Asian films by chaining actor/director biographical details — birth year, debut film, family, awards — to identify the person first, then find the movie
always: true
---

# Actor Characteristic Chain

## When to use
When the question describes an actor or director's personal details (birth year, debut era, family relations, number of awards) without naming them directly. Common in Indian/South Asian film questions.

## Technique
South Asian film questions frequently identify movies indirectly through actors' biographies. Extract the person's unique characteristics: birth year, debut film era, singing in debut, family relations, awards count.

Search for the person first using their most distinctive biographical detail. Once identified, search their filmography to find the target movie. Then verify all remaining constraints.

The chain is: biographical clues → person identity → filmography → target movie → verify constraints.

## Query Templates
- `"born [year]" [film industry] actor debut film [era]`
- `[actor name] filmography [year range]`
- `director born [year range] [film industry] [era] film`

## Worked Examples

### Example
- Question: 1960s film, director born 1925-1930, male lead born 1941 sang duet in debut film
- Search: `"born 1941" Bollywood actor debut film 1960s sang duet`
- Found: Pakistani actor Nadeem (born 1941)
- Verification: `Nadeem born 1941 Chakori debut film Pakistani cinema`
- Why it worked: The precise birth year + debut singing detail uniquely identified the actor

## Anti-pattern
- **Skipping person verification**: After finding a candidate actor, you must verify ALL personal characteristics (birth year, family, awards) before linking to a movie. A partial match can lead to the wrong film.
