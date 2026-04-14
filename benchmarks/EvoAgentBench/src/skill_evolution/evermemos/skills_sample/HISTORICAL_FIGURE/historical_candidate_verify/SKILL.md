---
name: historical_candidate_verify
description: >
  List candidate historical figures first, then search-verify each one against all question constraints
always: true
---

# Candidate Hypothesis and Verification for Historical Figures

## When to use
When initial searches return a candidate name, or when the clues strongly suggest a known historical figure and you can form a hypothesis before exhaustive searching.

## Technique
When a candidate name emerges, immediately switch to targeted verification: search "[person name] [specific detail]" for each constraint. All must match.

If no candidate emerges, use cultural context (Japanese immigrants, American frontiersmen, Indian poets) to narrow the field, then verify when a candidate appears.

Checklist: (1) Time period. (2) Location. (3) Family details. (4) Events/achievements. (5) Remaining constraints.

## Query Templates
- `[person name] [specific detail] [year]`
- `[person name] children siblings "third child" family`
- `[person name] [cultural keywords: internment/frontier/homeopathy]`

## Worked Examples

### Example
Candidate: Frank Hirata (from Hirata family, Japanese immigrant to Washington).
- Verify: `"Frank Hirata Kazuma returned Japan marry 1916 1917 1918 1919 1920"` — confirmed marriage trip
- Verify: `"Hirata farmer hotelier Washington"` — confirmed occupation
- Key: Expanded year ranges into specific year lists for better matching

### Example
Candidate: Samuel Hahnemann (identified via "clay candlestick" detail).
- Verify: `"Hahnemann childhood clay candlestick read books hiding"` — confirmed childhood story
- Verify: `"Hahnemann five children siblings third of five brothers sisters"` — confirmed birth order
- Key: Each constraint verified independently to avoid false positives

## Anti-pattern
**Continuing broad searches after a candidate appears**: Once you have a name, switch immediately to verification. Broad searches after candidate lock-in waste steps and may introduce confusion.
