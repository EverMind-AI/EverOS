---
name: geo_historical_monument
description: >
  Associate historical events with their corresponding monuments or memorials for location search
always: true
---

# Historical Event + Monument Association

## When to use
When the question involves a pattern of "a historical event led to the construction of a monument/memorial" or asks about a landmark tied to a specific historical event.

## Technique
Pattern: a historical event prompted the construction of a monument. Search for the event first, then find the corresponding monument.

Workflow: (1) Search "[city] [event description] [year]". (2) Once identified, search "[event name] monument memorial [city]". (3) Verify monument features (artist, year built) against the question.

Census data and administrative records ("2012 bulletin", "1901 syndicate purchase") are also strong search anchors.

## Query Templates
- `[city name] [event description] [year] memorial monument`
- `[event name] monument sculptor artist born [year]`
- `[location] [year] bulletin syndicate purchase [amount]`

## Worked Examples

### Example
Question: monument in a Bosnian city, artist born 1928, related to an event before 1970.
- Event search: `"Zenica Partisan Detachment May 1942 killed perished event"` — identified the WWII partisan event
- Monument search: then found the specific monument commemorating this event, verified artist birth year

### Example
Question: 2012 bulletin, Scottish syndicate purchased land for 5000 pounds in 1901, 53 houses by 1933.
- Search: `"Glencairn Simon's Town 2012 bulletin Scots syndicate 1901"` — the bulletin and syndicate details were distinctive anchors
- Then: `"Glencairn Cape Town factory 1902 glass furnace street name"` — located the specific street

## Anti-pattern
**Searching for the monument without understanding the event**: The event is usually more searchable than the monument itself. Always search for the historical event first, then find the associated memorial.
