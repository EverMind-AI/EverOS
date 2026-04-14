---
name: bio_taxonomic_combo
description: >
  Combine multiple taxonomic features (habitat + morphology + geography + behavior) to identify and locate target species
always: true
---

# Taxonomic Feature Combination for Biology

## When to use
When a question provides multiple biological features of a species — appearance, behavior, habitat, geographic range, cultural significance — and you need to identify the organism.

## Technique
Species identification questions typically provide 3-5 taxonomic features. Select the 2-3 most distinctive features and combine them in a single search. Geographic distribution + one unique behavioral trait is often enough to identify the species.

After identification, verify by checking that ALL features in the question match the candidate species. If not, try another combination.

## Country Identification from Demographics
When a question mentions country-level demographics (population growth projections, population density, endemic species), **identify the country first** before searching for the species. Key indicators:
- "population projected to grow 15-20% by 2050" → India (1.4B → ~1.6B)
- "endemic to a country" + large population → India, Brazil, Indonesia, China
- State/region population density → use specific state names (e.g., "Meghalaya population density" to confirm India)

This step often resolves the entire question: once the country is identified, "endangered species endemic to [country]" becomes a tractable search.

## Query Templates
- `[distinctive behavior] [geographic region] [habitat type] [morphological feature]`
- `[species name] described named [era] [naming author] naturalist`
- `[country name] endemic endangered [species type] [conservation status]`
- `"[state/region name]" population density [year] [value]`

## Worked Examples

### Example 1: African insect delicacy
- Question: A beloved insect in African countries, a local delicacy in northern South Africa, burrows underground to pupate
- Successful query: `insect delicacy northern South Africa burrowing underground pupate`
- Follow-up: `Mopane Worm harvesting project early 2010s protein source quoted`
- Why it worked: Behavioral + geographic features identified Mopane Worm; then used common name + specific clues to locate the article

### Example 2: Carnivorous plant endemic to India
- Question: Carnivorous species, endangered, endemic to a country, local name means "devil's basket"
- Step 1: Searched `"Meghalaya population density 2011 132 per sq km"` → confirmed India
- Step 2: Searched `"devil's basket" carnivorous species endemic India endangered` → found Nepenthes khasiana
- Why it worked: Identifying the country (India) first through demographic data drastically narrowed the species search space

## Anti-pattern
- Confusing taxonomic levels during search (e.g., searching Amanita when the target is Coprinus) — all features need to be verified for a match before proceeding.
- Skipping country identification when demographic clues are available — directly searching species features without narrowing by geography yields too many candidates.
