---
name: track_artist_first
description: >
  Find the artist first, then locate the specific track — track names alone are rarely distinctive enough for direct search.
always: true
---

# Music Track: Artist First, Track Second

## When to use
When a music track question describes both the artist and track characteristics.

## Technique
Track names are usually too generic for direct searching ("The Light", "Moving Forward", "Million Voices"). The effective approach is: identify the artist first, then search their discography for the matching track.

Search order:
1. Extract artist-level constraints: nationality, birth year, zodiac sign, career type (DJ, producer, singer), group affiliation.
2. Search for the artist using the most distinctive constraint.
3. Once you have the artist name, search `"[artist name] discography"` or `"[artist name] [album/track characteristics] [year]"`.
4. Verify the track matches all remaining constraints (release year, album position, lyrical theme).

For questions that describe album position ("third album", "sixth song on the tracklist"), after finding the artist, search their discography page directly to count albums or tracks.

## Query Templates
- `"[nationality] [music genre] [producer/DJ/singer] born [year range] [unique event]"`
- `"[artist name] discography albums"`
- `"[artist name] [album name] tracklist [track number]"`

## Worked Examples
**Million Voices:** Identified Swedish EDM scene from "EU member since 1995" + birth year → found Otto Knows → searched `"Otto Knows "Million Voices" 2012"` to confirm the track.

**Kune Rima:** Found Thomas Mapfumo through biographical clues → then located the specific track with `"Thomas Mapfumo album 2000s tracklist sixth song"`.

## Anti-pattern
Searching for the track title directly. Names like "The Light" or "Moving Forward" return millions of unrelated results. Always find the artist first to narrow the search space.
