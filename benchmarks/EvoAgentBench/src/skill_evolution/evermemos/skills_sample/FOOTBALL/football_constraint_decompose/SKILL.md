---
name: football_constraint_decompose
description: >
  Decompose multi-constraint football questions into incremental searches, starting from the most distinctive single constraint.
always: true
---

# Football: Constraint Decomposition

## When to use
When a football question has multiple layered constraints and you need to decide search order.

## Technique
Football questions typically stack 4-6 constraints. Searching all at once dilutes effectiveness. Instead, rank constraints by distinctiveness and search incrementally.

Priority order for constraints:
1. Specific match results ("1-0", "FA Cup final 1973") — most pinpointing
2. Historical milestones ("first FA Cup final at Wembley", "club founded 1932")
3. Player milestones ("retired August 2014", "first from his country in Premier League")
4. Year ranges for birth/career start — least distinctive, use last

After each search, extract candidate entities (club names, player names) and combine them with the next constraint for a follow-up search. This cascading approach narrows results step by step.

## Query Templates
- `"[year] [cup name] final [team A] [team B] [score]"`
- `"football club founded [year] [country] [cup achievement]"`
- `""announced his retirement" [month] [year] footballer"`

## Worked Examples
**Amr Zaki:** Question had birth year 1981-1984, career start 1999-2002, club founded 1930-1933, first FA Cup final at Wembley. Started with "FA Cup final Wembley" (most specific) → found Wigan Athletic → searched "Wigan Athletic player retirement August 2014" → found Amr Zaki.

**Sol Campbell:** Match between two top-tier teams in early 2000s, one founded in 1870s. Searched `"football match 2001 2002 substitute third away team"` → found Bolton vs Arsenal → verified from there.

## Anti-pattern
Searching all year ranges at once: "born 1981 1982 1983 1984 started career 1999 2000 2001 2002" — wide ranges dilute results. Pick the single most distinctive constraint first.
