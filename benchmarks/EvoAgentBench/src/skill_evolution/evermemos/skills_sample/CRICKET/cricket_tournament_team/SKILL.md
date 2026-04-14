---
name: cricket_tournament_team
description: >
  Lock down the tournament type and participating teams first to narrow the search space for cricket questions.
always: true
---

# Cricket: Tournament + Team Identification

## When to use
When a cricket question references a tournament (IPL, ICC, ODI, U-19 World Cup, Test) or provides team clues.

## Technique
Cricket is structured around tournaments and bilateral series. Identifying the tournament type and teams first dramatically narrows the search space before looking for specific match details.

Step 1: Determine match format — Is it ODI, T20, Test, IPL, or a specific ICC tournament? Clues include: "white ball" (limited overs), "five-day" (Test), franchise team names (IPL), "World Cup" (ICC event).

Step 2: Identify participating teams — Country names, IPL franchise names, or indirect clues ("non-mainstream cricket nation" could be PNG, Zimbabwe, Nepal).

Step 3: Combine tournament + teams + any scoring data for a focused search.

Important: Do not assume only India/Australia/England. Questions frequently involve Papua New Guinea, Zimbabwe, and other associate/affiliate nations.

## Query Templates
- `"IPL [year] [team A] [team B] [key player]"`
- `"[team A] v [team B] [year] [tournament name] [score/margin]"`
- `"Under-19 World Cup [year] highest wicket taker [team name]"`
- `"[team A] vs [team B] [year] "Player of the Match" [key data]"`

## Worked Examples
**ODI no. 1880:** Question described an ODI before 2006 with wicket-keeper injury. Searched `"2002 ICC Champions Trophy match wicket-keeper injury player of match"` → found England v Zimbabwe match directly.

**Norman Vanua:** Cricket player from a non-mainstream nation. Searched `"USA vs PNG ODI April 27 2019 164 runs Player of the Match"` → hit on the specific match involving Papua New Guinea.

## Anti-pattern
Ignoring lesser-known teams. Cricket questions may involve PNG, Zimbabwe, Nepal, or other non-Test nations. Do not default to major cricket countries only.
