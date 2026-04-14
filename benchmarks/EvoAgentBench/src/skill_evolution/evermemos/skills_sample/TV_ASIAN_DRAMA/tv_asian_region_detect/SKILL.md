---
name: tv_asian_region_detect
description: >
  Determine whether the target drama is Korean, Turkish, Indian, or Thai from cultural clues in the question before searching
always: true
---

# Cultural Region Detection

## When to use
When the question describes a non-English TV drama but does not explicitly state the country of origin. Clues about language, cultural context, actor names, or broadcast channels can reveal the region.

## Technique
Before searching, analyze the question for regional indicators: actor name styles (Korean vs Turkish vs Indian), cultural themes (arranged marriage → Indian, historical palace → Korean or Turkish), broadcast channels, and streaming platforms.

Turkish dramas often mention multiple TV channels, alternative language titles, and themes of family honor. Korean dramas mention Hallyu themes, multiple alternative titles, and specific K-drama tropes. Indian dramas mention streaming platforms and music/dance themes.

Once the region is identified, use region-specific search keywords (e.g., "Korean drama", "Turkish dizi", "Indian web series") to dramatically narrow results.

## Query Templates
- `Korean drama [plot keywords] [year range]`
- `Turkish TV series [theme] [era] [channel]`
- `Indian drama series [year] streaming [platform]`

## Worked Examples

### Example
- Question: 1990-2005 TV series, female lead acted in 1980s stage plays, aired on two TV channels
- Region detection: "Two television channels" + era → Turkish drama pattern
- Search: `Turkish TV series comedy 1990s 2000s two television channels`
- Why it worked: Correctly identifying the Turkish origin let us use "Turkish TV series" as a precise filter

## Anti-pattern
- **Ignoring regional clues and searching generically**: `TV series female lead stage play 1990s` returns Western shows. Always detect the cultural region first and include it in searches.
