---
name: actor_indian_bollywood_kw
description: >
  Always add Bollywood/Hindi film/Indian actor keywords when searching for Indian actors
always: true
---

# Bollywood Keywords for Indian Actor Search

## When to use
When you determine the target person is an Indian or South Asian actor, and you are constructing any search query related to them.

## Technique
Searches for Indian actors without domain keywords return results overwhelmed by Western actors. Every query must include at least one of: "Bollywood", "Hindi film", "Indian actor", "Indian actress", or specific Indian award names like "Dadasaheb Phalke", "National Film Award", "Filmfare Award".

Indian film awards are extremely powerful domain qualifiers. If the question mentions any award, use the specific award name. Even without award mentions, adding "Bollywood" or "Hindi film" dramatically improves precision.

For some actors, Hindi Wikipedia contains richer biographical information than English Wikipedia. Consider searching in Hindi if English results are insufficient.

## Query Templates
- `[event description] actor Bollywood Hindi film`
- `[actor name] "Dadasaheb Phalke" OR "National Film Award" OR "Filmfare Award"`
- `[actor name] controversy [year] [event keywords] Bollywood`

## Worked Examples

### Example
Question mentions: father is an actor (born 1940-1960), father married actress (1970-1990), interview biography book about father.
- Search: `"actor award \"Dadasaheb Phalke\" \"National Film Award\" born 1920 1930"` — Indian awards immediately narrowed scope to Bollywood
- Then: `"Amitabh Bachchan" book "interviews" biography` — verified after identifying father

## Anti-pattern
**Searching without domain keywords**: `"actor flight incident 2006"` returns too many irrelevant Western results. Must use `"actor flight incident 2006 Bollywood"` or `"Indian actor flight incident 2006"`.
