---
name: cricket_scorecard
description: >
  Use precise scoring data (runs, wickets, overs, balls faced) as search anchors for cricket match identification.
always: true
---

# Cricket: Scorecard Search

## When to use
When a cricket question contains specific numerical scoring data — runs scored, wickets taken, overs bowled, balls faced, margin of victory, or detailed batting/bowling figures.

## Technique
Cricket questions are rich in numerical data that maps directly to scorecard databases (ESPN Cricinfo, etc.). Numbers are the most powerful pinpointing tool in cricket search because exact scores are unique identifiers.

Combine scoring numbers with match type and year to construct scorecard-style queries. For example, "168/9 IPL 2009" will match far fewer results than "IPL 2009 match."

Key number types to exploit:
- Team totals: "168/9", "164 all out"
- Individual scores: "20 runs off 19 balls"
- Bowling figures: "7 wickets in last 7 overs"
- Victory margins: "won by over 100 runs"
- Over-specific events: "wicket-keeper replaced after 5.3 overs"

When numbers alone are not enough, add the match type (ODI/T20/Test/IPL) and any team name clues.

## Query Templates
- `"[team name] [total]/[wickets] [match type] [year] batting scorecard"`
- `"cricket match last seven overs [runs] runs [wickets] wickets"`
- `"[team A] v [team B] [year] [score] [tournament]"`

## Worked Examples
**Pragyan Ojha:** Question described 41-43 runs and 7 wickets in last seven overs. Searched `"cricket match last seven overs 41 42 43 runs 7 wickets"` → direct hit on the IPL 2009 match → found Player of the Match.

**Mohammad Yousuf:** Player scored 20 runs off 19 balls on debut. Searched `"cricketer born Lahore debut 2000 2010 T20 20 runs 19 balls"` → found the player directly from debut stats.

## Anti-pattern
Broad searches without numbers: "ODI match before 2006 won by 100 runs" is too vague. Always include the specific scoring numbers from the question.
