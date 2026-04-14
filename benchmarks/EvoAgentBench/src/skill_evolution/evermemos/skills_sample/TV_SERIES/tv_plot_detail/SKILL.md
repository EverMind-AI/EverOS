---
name: tv_plot_detail
description: >
  Combine multiple specific plot details (character events, season incidents) into a single search — this yields extremely high hit rates for TV series
always: true
---

# Plot Detail Combination Search

## When to use
When the question describes specific plot events tied to characters or seasons (e.g., "roommate sacrifices himself in S3 finale", "character suffers an injury the actor experienced in real life").

## Technique
TV series questions often contain highly specific plot descriptions that are almost unique fingerprints. Combine 2-3 of the most distinctive plot details into one search query. These combinations are far more effective than generic genre or era searches.

Extract the most unusual plot elements: character-specific events, season/episode references, real-life actor connections to plot. Combine them with "TV series" as an anchor term.

Once a candidate show is found, immediately verify by searching for the remaining plot constraints with the show name.

## Query Templates
- `[distinctive plot detail 1] [detail 2] TV series`
- `male lead roommate sacrifice season [N] finale TV series`
- `"producer credit" actor injury suggested plot writer`

## Worked Examples

### Example
- Question: Show where male lead's roommate sacrifices in S3 finale, female lead goes home to have baby in S4
- Search: `male lead roommate sacrifice season 3 finale TV series`
- Result: Directly hit "You're the Worst"
- Why it worked: "roommate sacrifice season 3 finale" is an extremely distinctive combination

## Anti-pattern
- **Searching with only genre and era**: `drama TV series 2010s character development` returns thousands of results. Always use specific plot details instead.
