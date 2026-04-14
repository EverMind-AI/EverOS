---
name: crime_case_identify
description: >
  First determine the case type (murder, theft, arson, accident), then search using domain-specific terminology
always: true
---

# Case Type Identification for Crime Events

## When to use
When the question describes a criminal or legal case and you need to identify the specific incident before answering detailed questions about it.

## Technique
First classify the case type from the question description: murder/homicide, theft/robbery, arson/fire, traffic accident, assault, fraud, etc. Each type has domain-specific terminology that dramatically improves search precision.

Type-specific keywords:
- Murder: homicide, murdered, killed, trial, defendant, convicted
- Theft/Robbery: stolen, heist, robbery, ATM, malware, looted
- Fire/Arson: fire, arson, blaze, victims, arrested, charged
- Traffic accident: vehicular accident, car crash, collision, fatality

Combine the case type keyword with the most distinctive constraint (amount stolen, number of victims, specific location, victim profile) for the initial search.

## Query Templates
- `[case type] [victim characteristics] [time range] [location]`
- `[case type keyword] [distinctive amount/number] [year range]`
- `[specific detail] [case type] [country/city]`

## Worked Examples

### Example
Question: bank ATMs robbed of $200K-$400K, 15-35 ATMs, 2010-2023.
- Case type: theft/robbery with malware
- Search: `"ATM malware hacking bank stolen $200000 $400000 automated teller machines shutdown"` — the amount + ATM count was distinctive enough

### Example
Question: fire at educational institution 2009-2021, minors arrested, believed to be misunderstanding.
- Case type: arson/fire at school
- Search: `"school fire arrested teenagers misunderstanding students 2009 2021"` — located the Malaysia tahfiz school fire

## Anti-pattern
**Searching without case type keywords**: Generic searches like "incident 2016 bank" are far less effective than "ATM malware theft 2016 bank". Always include the case type term.
