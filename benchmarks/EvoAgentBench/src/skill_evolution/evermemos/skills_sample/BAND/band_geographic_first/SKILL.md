---
name: band_geographic_first
description: >
  Decode indirect geographic clues (area, EU membership, flag colors) to identify the country before searching for the band.
always: true
---

# Band: Geographic Identification First

## When to use
When a band question describes the country indirectly through land area, EU membership, flag colors, or population.

## Technique
Band questions frequently encode the country as a puzzle: "area between 700-800 sq km", "EU member since 1995", "celebrated independence in [year]". Solving this geographic clue first is critical because it narrows the search space from all bands worldwide to bands from one country.

Steps:
1. Extract the geographic clue and solve it. Use web search if needed: `"country area 700 square kilometers"`.
2. Be aware that multiple countries may fit: 700-800 sq km could be Singapore (728) or Bahrain (765). Prepare to try multiple candidates.
3. Once you have the country, search `"[country] [music style] band formed [year]"`.
4. If the first country yields no results, try the next candidate country.

Common indirect geographic clues:
- Land area ranges → look up countries by area
- EU membership year → specific set of countries per accession year
- Independence year → narrows to specific countries
- Flag color descriptions → cross-reference with flag databases

## Query Templates
- `"country area [N] square kilometers"`
- `"[country] [rock/jazz/folk] band formed [year]"`
- `"EU member since [year]" countries list`

## Worked Examples
**Duman:** "Country with area 700-800 sq km" → tried Singapore, then Bahrain → eventually found Turkish band Duman through `""folk rock" band 1999 bassist Cancer zodiac"`. The area clue required trying multiple country candidates.

## Anti-pattern
Guessing the country without verifying. "700-800 sq km" has multiple valid answers — you must try each candidate systematically rather than assuming the first one is correct.
