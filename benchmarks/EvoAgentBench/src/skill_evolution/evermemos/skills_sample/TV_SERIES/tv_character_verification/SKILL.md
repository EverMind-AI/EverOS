---
name: tv_character_verification
description: >
  After finding a candidate TV series, verify it by searching character name + show name together to confirm plot constraints
always: true
---

# Character Name + Show Name Verification

## When to use
When you have a candidate TV series and need to confirm it matches all the question's constraints — character details, birth years, episode appearances, relationships.

## Technique
Once a candidate show is identified, do not assume it is correct. Search `"[show name]" character [specific detail]` to verify each constraint. Character names, birth years, episode counts, and relationship details can all be verified this way.

This is the final verification step. Check constraints one by one: character birth era, number of episode appearances, specific plot events, actor real-life connections.

If any constraint fails verification, reject the candidate and search for alternatives.

## Query Templates
- `"[show name]" character [character name] born [year]`
- `"[show name]" [character] episodes appeared season [N]`
- `[actor name] [real experience] [show name] plot suggested`

## Worked Examples

### Example
- Question: Which season of a TV show features a character suffering an injury the actor survived in real life?
- Found candidate: The Good Karma Hospital, Amanda Redman
- Verification: `Amanda Redman burns injury 1950s child charity advocate`
- Why it worked: Verified the actor's real-life burn injury matched the character's plot, confirming the show and season

## Anti-pattern
- **Accepting the first candidate without verification**: Finding a show that matches one constraint and assuming it matches all. Always verify every constraint with targeted searches.
