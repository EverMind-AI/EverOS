---
name: crime_non_english
description: >
  For cases in non-English countries, use local language terms and local media terminology
always: true
---

# Non-English Case Awareness for Crime Events

## When to use
When clues suggest the crime occurred in a non-English-speaking country (Thailand, Malaysia, Philippines, Latin America, etc.) and English searches are not returning relevant results.

## Technique
Cases in non-English countries are often reported primarily in local media using local terminology. English searches may miss these entirely. When you identify the country, incorporate local-language terms into your searches.

Key local terms by region:
- Malaysia: "tahfiz" (Islamic scripture school), Malay place names
- Thailand: "Sukhumvit", Thai bank names (GSB, Krungsri), Thai district names
- Philippines: Filipino names, local TV networks (ABS-CBN, GMA)
- Latin America: Spanish legal terms, local newspaper names

Also be aware of currency differences: the question may state amounts in USD, but local reports use local currency.

## Query Templates
- `[local term] [event type] [year] [country]`
- `[local place name] [case type] [specific detail]`
- `"[local street/district name]" [case detail] [year]`

## Worked Examples

### Thailand ATM theft
- English search found the case, but specific location needed Thai terms
- Search: `"Sukhumvit Soi 23" second ATM Bangkok GSB hacking order` — Thai street name was essential

### Malaysia school fire
- Key breakthrough: `"Malaysia tahfiz fire 23 victims buried funeral September 2017"`
- "Tahfiz" was the critical local term that English "school" couldn't capture

## Anti-pattern
**Using only English generic terms**: Searching "school fire Malaysia" is far less effective than "tahfiz fire Malaysia". Local terminology is often the key to finding the right case.
