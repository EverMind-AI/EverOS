---
name: game_technical_detail
description: >
  Use game mechanics, platform (DOS/Atari/NES), distribution method (shareware), engine, and review scores as search clues to identify games
always: true
---

# Technical Detail Search

## When to use
When the question mentions specific platforms (DOS, Atari, NES, PS4), distribution methods (shareware, floppy disk), review scores from specific sites (JeuxVideo), or technical details like game engines.

## Technique
Technical details are highly discriminative for video games. Platform + era + distribution method combinations dramatically narrow the search space. Review scores from specific sites (e.g., "JeuxVideo 90%") are nearly unique identifiers.

Search with the platform and era as primary anchors, combined with one gameplay or distribution detail. For review-based clues, search the specific review site directly.

Credit roles (illustrator, animator, planner) in game development are also technical details — search `"[game name]" credits [role]` to find specific contributors.

## Query Templates
- `[game name] [platform] DOS shareware [year]`
- `JeuxVideo [rating] [era] [game genre]`
- `[game name] credits [specific role] illustrator animator`

## Worked Examples

### Example
- Question: 1980s game rated 90% on JeuxVideo, illustrator born in a specific year
- Search: `Japanese video game illustrator 1980s PS4 2017 planner birth 1960 1970 1980`
- Found: Masato Kato linked to the target game
- Why it worked: Combined "illustrator" + era + platform constraints to locate the specific person and game

## Anti-pattern
- **Ignoring platform and distribution details**: Searching `1990s adventure game puzzle` when you know it was DOS shareware. Platform details are among the most powerful filters for older games.
