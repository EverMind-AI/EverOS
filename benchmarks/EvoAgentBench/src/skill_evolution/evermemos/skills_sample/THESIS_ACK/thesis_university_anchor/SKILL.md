---
name: thesis_university_anchor
description: >
  Use university name + degree type + year combinations to precisely locate theses, resolving indirect university clues first
always: true
---

# University Anchoring for Theses

## When to use
When a question provides university clues — either directly (university name) or indirectly (e.g., "founded the same year as a novel", "a university in California with X applicants").

## Technique
University + degree type + year is a powerful combination for locating theses. If the university is described indirectly, resolve it first. For example, "founded the same year as a novel narrated from a boy's perspective" requires identifying the novel and its publication year to determine the university's founding year.

Always resolve indirect clues before searching for the thesis itself. This two-step approach is far more effective than guessing.

## Query Templates
- `university founded [year] [country/region]`
- `"[university name]" [discipline] [degree type] [year] thesis`

## Worked Examples

### Example
- Question: A PhD thesis submitted to a department of a university that was founded in the same year as a novel narrated from a boy's perspective
- Step 1: `Adventures of Huckleberry Finn published 1885 USA` (identify the novel and year)
- Step 2: `"multiethnic eastern province" dissertation PhD European` (search for the thesis)
- Why it worked: First resolved the indirect clue (boy's narrative novel = Huck Finn, 1885) to determine the university's founding year, then searched for the thesis

## Anti-pattern
Not resolving indirect clues — skipping the university identification step and searching for the thesis directly will fail when the university is described indirectly.
