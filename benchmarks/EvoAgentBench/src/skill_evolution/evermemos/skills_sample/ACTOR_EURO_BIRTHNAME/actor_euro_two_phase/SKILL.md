---
name: actor_euro_two_phase
description: >
  Two-phase search — first find the actor's stage name, then specifically search for their full birth name
always: true
---

# Two-Phase Search for European Actor Birth Names

## When to use
When the question asks for the full birth name of a European actor, especially French actors who commonly use stage names different from their legal birth names.

## Technique
Phase 1: Locate the actor's stage name through clues (film works, era, birthplace, spouse, awards).

Phase 2: Once the stage name is known, do NOT assume it is the birth name. Search specifically for "birth name", "née", "born as", or "nom de naissance". European actors frequently have multiple middle names or entirely different legal names.

If English sources lack the birth name, escalate to French Wikipedia.

## Query Templates
- Phase 1: `[director name] [film genre] film [era] actor actress`
- Phase 2: `[actor stage name] birth name full name née`
- Phase 2 fallback: `[actor stage name] nom de naissance Wikipedia`

## Worked Examples

### Example
Question: born 1920s Paris, 1950s comedy film, director died near Cannes 1986-1995.
- Phase 1: `"French comedy film 1950 1951 1952 1953 1954 1955 actor born 1920s Paris"` — identified the actress
- Phase 2: `"Jacqueline Delubac born Paris actor family grandfather"` — searched for full birth name details

### Example
Question: born 1930s France, first spouse died of cancer 1995-2005, sibling is musician.
- Phase 1: `"French person born 1930s died 2000s spouse cancer 1995 2005"` — identified Pascale Audret
- Phase 2: `"Pascale Audret birth name nom de naissance"` — retrieved full legal name

## Anti-pattern
**Stopping at the stage name**: After finding the common name, assuming it is the birth name without a dedicated search. French actors' full birth names (with multiple middle names) are often only found through explicit "birth name" queries.
