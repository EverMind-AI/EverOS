---
name: bio_chemical_numeric
description: >
  Use precise chemical compound names and concentration values as exact-match search anchors for biology papers
always: true
---

# Chemical/Numerical Exact Matching for Biology

## When to use
When a question mentions specific chemical compounds, concentration values (e.g., "0.330 mg/100g"), synonym counts, or other precise numerical data related to a biological species or study.

## Technique
Chemical compound concentrations and precise numerical values are extremely strong discriminators in biology searches. Use quotation marks around exact values and compound names. Combining 2-3 chemical names or a compound name + concentration value can uniquely identify a paper or species.

Also leverage synonym counts from taxonomic databases — "5 unaccepted synonyms" combined with a species type can be very distinctive.

## Query Templates
- `"[chemical compound name]" "[concentration value]" [species name] [method]`
- `"[compound 1]" "[compound 2]" [species type] edible`

## Worked Examples

### Example
- Question: A species named in the 1780s, 5 unaccepted synonyms, containing specific flavonoid compound concentrations
- Successful query: `"Epicatechin" "Catechin" "Amentoflavone" mushroom edible`
- Why it worked: Combined chemical compound names (flavonoids) + species type (mushroom) for search, precisely locating Coprinus comatus

## Anti-pattern
Not using quotation marks for exact matching when the question provides specific compound concentrations — wasting the strongest discriminator available.
