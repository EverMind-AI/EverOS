---
name: tv_asian_platform
description: >
  Use streaming platform and broadcast channel information (Viki, Netflix, Amazon Prime, specific TV channels) to narrow down Asian drama candidates
always: true
---

# Streaming Platform Clues

## When to use
When the question mentions streaming platforms (Amazon Prime, Netflix, Viki), broadcast channels, or "streaming" as a distribution method. Platform information combined with year and genre is a powerful filter.

## Technique
Streaming platform details significantly narrow the search space. "Streaming one season" + year range + genre often identifies a show directly. Different platforms have different regional strengths: Viki for K-dramas, Amazon Prime for Indian web series, Netflix for Turkish dramas.

Combine platform with broadcast year, season count, and one plot keyword for an efficient first search. Channel information ("aired on two TV channels") is equally useful for older dramas.

## Query Templates
- `[genre] drama series [year] streaming [platform] one season`
- `[theme] series Amazon Prime [year range]`
- `drama [year] [channel name] broadcast Turkish Korean Indian`

## Worked Examples

### Example
- Question: Music-themed series, early 2020s streaming, lead actor has multiple awards and a memoir
- Search: `music drama series 2020 2021 2022 streaming one season`
- Result: Indian drama "Bandish Bandits" on Amazon Prime
- Verification: `Bandish Bandits cast actors memoir awards`
- Why it worked: "Streaming" + "one season" + "music" + year range was enough to identify the show

## Anti-pattern
- **Ignoring platform information**: When the question says "streaming on a platform", not using that as a search filter wastes a strong discriminative clue.
