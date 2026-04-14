---
name: geo_surrounding_cross
description: >
  Use surrounding geographic features (rivers, businesses, nearby buildings) to cross-locate a specific landmark
always: true
---

# Surrounding Environment Cross-Location

## When to use
When the question mentions commercial facilities, natural features, or other landmarks near the target location (e.g., restaurants, law firms, opticians, rivers, mountains).

## Technique
The combination of multiple nearby businesses is highly unique to a specific street or square.

Workflow: (1) Identify the general area using other clues. (2) Search "[landmark or area] [business type 1] [business type 2]". (3) The overlap pinpoints the exact location.

For distance constraints, always verify by searching "[location A] to [location B] distance miles" — never estimate by intuition.

## Query Templates
- `[landmark name/street name] [surrounding business names] restaurant optician solicitors`
- `[landmark name] [nearby natural feature] river mountain`
- `[location A] to [location B] distance miles aerial`

## Worked Examples

### Example
Question: European landmark unveiled early 20th century, renovated early 21st century, nearby: pizza shop, law firm, fish shop, Asian fusion restaurant, optician.
- First located the city (Wexford), then: `"Bull Ring Wexford pizza restaurant solicitors optician Asian fusion"` — the combination of five surrounding businesses uniquely identified the location
- Key: No single business is distinctive, but the combination of all five is unique

### Example
Question: Irish water structure, built 19th/early 20th century, used for stability, nearby restaurant/hotel named after it.
- Search: `"Ballast Bank Wexford restaurant hotel named after"` — the naming relationship confirmed the location

## Anti-pattern
**Ignoring surrounding business clues**: Names and types of nearby restaurants and shops are excellent verification keywords. Do not skip them — they often provide the final confirmation.
