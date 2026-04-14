---
name: football_lead_tracking
description: >
  Track every player or club name that appears in search results — never ignore a lead, always follow up.
always: true
---

# Football: Lead Tracking

## When to use
Every time a search returns results containing player names, club names, or match references that could be relevant to the question. This is a discipline to apply throughout the search process.

## Technique
Any person name or club name that appears in search results must be tracked and searched further. The most common failure mode in football questions is finding the right entity in results but not following up.

Tracking protocol:
1. After every search, scan results for entity names (players, clubs, coaches).
2. For each new entity, immediately search `"[entity name] [most distinctive remaining constraint]"`.
3. If the entity satisfies one constraint, keep searching to verify the rest.
4. Maintain a mental list of all candidate entities found so far — do not lose track of earlier leads when pursuing new ones.

This applies especially to intermediate entities: a club name might lead to a player, or a match record might lead to a coach. Follow the chain.

## Query Templates
- `"[newly found name] [next constraint to verify]"`
- `"[club name] player [event from question]"`
- `"[player name] [club name] [year] [specific match detail]"`

## Worked Examples
**Sol Campbell:** Searched match details → found "Bolton vs Arsenal" → tracked both team names → searched `"2001-02 Premier League table Arsenal Bolton wins"` → identified the specific match and player.

**Amr Zaki:** Found "Wigan Athletic" from FA Cup search → tracked the club name → searched `"Wigan Athletic player retirement August 2014"` → found Amr Zaki.

## Anti-pattern
Finding a candidate club name in results but then re-combining original question descriptions for a new search instead of using the found entity name. Always use discovered names as the next search keyword.
