---
name: cricket_career_verify
description: >
  After finding a candidate cricket player, verify all constraints using their career data from Wikipedia or Cricinfo profiles.
always: true
---

# Cricket: Player Career Verification

## When to use
When you have a candidate cricket player and need to confirm they match all constraints.

## Technique
Cricket player profiles on Wikipedia and ESPN Cricinfo contain structured data: birth date, debut date, debut score, career statistics, teams played for, and records held. After finding a candidate, pull up their full profile and verify each constraint.

Verification checklist:
1. Birth date and birthplace match the question's year/city clues
2. Debut details (format, opponent, score, balls faced) match
3. Career milestones (records, "first to X runs") match
4. Teams played for match the club/franchise clues
5. Player of the Match awards match specific games referenced

Also verify ODI/T20I/Test distinctions carefully — these are different formats with separate debut records. Confusing them is a common error.

## Query Templates
- `"[player name] Wikipedia birth date debut ODI T20I Test"`
- `"[player name] [match type] debut [score] runs [balls] balls [opponent]"`
- `"[player name] ESPN Cricinfo profile career statistics"`

## Worked Examples
**Mohammad Yousuf:** Found candidate from debut stats search. Verified with `"Mohammad Yousuf T20 debut Bristol 2006 20 runs 4s 6s boundaries"` — confirmed birth date, debut details, and match venue all matched.

**Pragyan Ojha:** After identifying the IPL match, verified with `"IPL 2009 match 12 Deccan Chargers Mumbai Indians "Player of the Match" Rohit Sharma Pragyan Ojha"` — confirmed the Player of the Match.

## Anti-pattern
Confusing match types: ODI and T20I debuts are different records. Always specify the exact format (ODI/T20I/Test) when verifying debut data. A player's T20I debut stats will not appear in ODI search results.
