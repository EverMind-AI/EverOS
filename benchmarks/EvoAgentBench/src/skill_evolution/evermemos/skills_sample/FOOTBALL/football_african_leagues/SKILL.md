---
name: football_african_leagues
description: >
  Search strategies for non-European football contexts including African, South Asian, Russian, and other non-mainstream leagues.
always: true
---

# Football: Non-European & Non-Mainstream Leagues

## When to use
When a football question involves clues pointing outside major European leagues — African clubs, South Asian leagues, Russian football, MLS, or any non-Premier League / La Liga / Bundesliga context. **Always consider non-European possibilities before committing to a European league search.**

## Technique
The football domain extends far beyond Europe. Default assumptions of "Premier League" or "European" will miss critical contexts.

### African Football
- Club name changes are common (Mimosa FC → FC Platinum)
- Use "CAF Champions League", "AFCON", "African Goal of the Year" as search terms
- South African football: Soweto Derby, PSL (Premier Soccer League), Bafana Bafana
- Nigerian football: NPFL (Nigeria Professional Football League), Rangers International
- When a question mentions awards related to African football, search `"African Goal of the Year"` or `"CAF award"` directly

### Indian Football
- I-League, ISL (Indian Super League) are the main leagues
- Search with `"I-League"` or `"ISL"` and player name
- Indian football certifications: AFC Pro license, AFC coaching courses
- Indian national team has distinctive constraints (30+ caps, specific tournaments)

### Russian Football
- Russian Premier League (RPL), formerly RFPL
- Clubs: FC Krasnodar, Spartak Moscow, Zenit, CSKA
- Trace through European competitions: Europa League, Champions League qualifying rounds

### First-from-country Players
- Many questions ask about "first [nationality] player in [league]"
- Search `"first [nationality] Premier League"` or `"first [country] player [league]"`
- Common pattern: African-born player with European nationality (e.g., Somali-born Norwegian)

## Query Templates
- `"[club name] [country] football [key event]"`
- `"CAF Champions League [year] [club name]"` / `"African Goal of the Year [year]"`
- `"I-League [player name] [year]"` / `"ISL [team] [year] squad"`
- `"first [nationality] player [league]"` / `"first [country]-born [league]"`
- `"[club name] Europa League [year]"` (for Russian/non-mainstream European clubs)

## Worked Examples

**Norman Mapeza, Zimbabwe:** Question described a coach announced in December 2023 for a team originally a social football club. Searched `"fourth successive league title fourth overall coach third title"` → found FC Platinum (formerly Mimosa FC) → verified with `"Norman Mapeza December 2023 coach announced"`.

**Abdisalam Ibrahim, Somalia/Norway:** Question asked for the first player from his birth country in the English Premier League. Searched `"first player from African country Premier League European nationality"` → narrowed to Somali-born → `"first Somali-born Premier League January 2014 Norwegian nationality"` → found Abdisalam Ibrahim.

**Indian football:** Question about an Indian player with 30+ caps and AFC Pro license. Searched `"Indian football player AFC PRO license 30 caps I-League"` → identified candidate → verified via `"football training session September school students Indian footballer"`.

**Benni McCarthy, South Africa:** Question about an African player who was the first from his country to win UEFA Champions League. Searched `"African football player first win UEFA Champions League"` → `"African Goal of the Year award"` → found Benni McCarthy → verified via `"ESPN Africa article 2017 AmaZulu"`.

**FC Krasnodar, Russia:** Question involved a 2019 England match incident. Traced the specific player via match details → found Europa League connection → `"Joe Williams Everton Europa League Krasnodar bench November 2014"`.

## Anti-pattern
Assuming all football questions involve the top 5 European leagues. When constraints don't match known Premier League / La Liga / Bundesliga data, immediately branch to: (1) African football, (2) Indian football, (3) Russian football, (4) MLS/South American football.
