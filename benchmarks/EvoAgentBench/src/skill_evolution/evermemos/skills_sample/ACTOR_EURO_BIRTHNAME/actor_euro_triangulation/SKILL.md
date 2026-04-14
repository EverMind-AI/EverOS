---
name: actor_euro_triangulation
description: >
  Director-Film-Actor triangulation — locate the target actor through the director's other works
always: true
---

# Director-Film-Actor Triangulation

## When to use
When the question describes an actor indirectly through a director's characteristics and film characteristics, rather than naming the actor.

## Technique
The question describes a director (often through death details), a film (era/genre), and the actor (birth details). Resolve each vertex in order:

(1) Locate the director using distinctive traits. (2) Search the director's filmography for the matching film. (3) Find the actor matching birth constraints from the cast. (4) Search for the actor's birth name.

Essential when the actor is obscure — the director or film serves as the entry point.

## Query Templates
- `[director characteristics] director died [place] [year range]`
- `[director name] filmography [film genre] [era]`
- `[film name] [year] cast actor actress born [birthplace] [birth era]`

## Worked Examples

### Example
Question: actress born 1920s Paris, appeared in 1950s comedy, director died near Cannes 1986-1995.
- Step 1 — Director: Searched for French directors who died near Cannes in that period
- Step 2 — Film: Searched director's filmography for 1950s comedies
- Step 3 — Actor: Found the actress born in 1920s Paris from the cast list
- Key: The director's death location (Cannes) was the most distinctive entry point

### Example
Question: actress born 1930s France, first spouse died of cancer 1995-2005, brother is musician.
- Verified through film: `"La Fayette 1961 film director Pascale Audret"` — cross-referenced the actress against a known film
- Key: Film works serve as independent verification of the actor's identity

## Anti-pattern
**Trying to find the actor directly**: When the actor is obscure, direct searches fail. Use the more famous director or well-known film as the entry point for triangulation.
