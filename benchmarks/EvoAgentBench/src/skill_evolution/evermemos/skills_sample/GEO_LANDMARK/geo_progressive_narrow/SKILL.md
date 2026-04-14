---
name: geo_progressive_narrow
description: >
  Progressively narrow from country to province/state to city to specific landmark
always: true
---

# Progressive Geographic Narrowing

## When to use
When the question asks about a specific geographic location or landmark, and provides clues about the broader region (country, administrative division, census data, historical context).

## Technique
Narrow geographic scope step by step:

1. **Country/Region**: Use historical events, cultural markers, administrative clues (e.g., "former Yugoslavia" -> Bosnia)
2. **City/Town**: Use census data, population figures, "largest cities" clues
3. **Specific Landmark**: Search "[city] monument/structure [features]"
4. **Verify**: Cross-check with remaining constraints

If the initial assumption contradicts other clues, correct the scope immediately.

## Query Templates
- `[country/region characteristics] largest cities [census year] population`
- `[city name] monument museum landmark [feature description] [year]`
- `museum established [year] renamed [year]`

## Worked Examples

### Example
Question: one of four largest cities in former Yugoslavia country, monument built before 1970, artist born 1928.
- Step 1: `"Bosnia largest cities 2013 census population"` — determined four cities: Sarajevo, Banja Luka, Tuzla, Zenica
- Step 2: `"monument artist born 1928 Bosnia Sarajevo Banja Luka Tuzla Zenica"` — narrowed to Zenica
- Step 3: `"Zenica Partisan Detachment May 1942 killed perished event"` — found specific monument

### Example
Question: museum established 1910, renamed 1999, ~602 miles from an aquarium.
- Step 1: `"museum established 1910 renamed 1999"` — directly located the museum
- Step 2: `"Fort Smith Arkansas to Newport Kentucky distance miles"` — verified distance constraint

## Anti-pattern
**Not correcting wrong geographic scope**: If your initially assumed location contradicts population data or other constraints, you must revise the assumption rather than ignoring the mismatch.
