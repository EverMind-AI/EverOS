---
name: game_person_story
description: >
  Locate games indirectly by searching for developers' personal stories — interviews, lifestyle details, hardware history — using distinctive phrases as exact-match queries
always: true
---

# Person Story Tracking

## When to use
When the question describes a developer's personal life: hardware their father bought, where they live, or distinctive lifestyle choices.

## Technique
Developers' unique personal stories appear in interviews, blogs, and talks. Search for distinctive phrases with exact-match quotes: `"hole in the map"`, `"solar-powered refrigerator"`. These phrases typically appear in only one or two sources, directly identifying the developer. Then trace to their games through their portfolio or company.

## Query Templates
- `"[distinctive phrase]" developer [lifestyle keyword]`
- `[developer name] [hardware name] father [era] purchased`
- `"[specific date]" gaming developer interview [detail]`

## Worked Examples

### Example
- Question: Developer who claims to have designed a solar-powered refrigerator and lives in a "hole in the map"
- Search: `Joey Hess rustic lifestyle "hole in the map"`
- Found: Developer Joey Hess directly
- Verification: `Joey Hess father computer Atari 130XE purchased`
- Why it worked: "Hole in the map" is an extremely distinctive phrase appearing in very few sources

### Example
- Question: Developer drew mammal character intro/end animations, found via specific date interview
- Search: `"July 25 2013" gaming PC wireless keyboard animated 8x11 paper`
- Found: Dean Dodrill
- Why it worked: The specific date + "8x11 paper" uniquely identified the interview

## Anti-pattern
- **Searching only game databases**: Developer interviews, personal blogs, and conference talks contain the biographical details needed. Standard game databases (IGDB, MobyGames) rarely include personal lifestyle information.
