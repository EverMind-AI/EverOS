---
name: musician_life_event
description: >
  Use unique personal life events (not musical works) as the primary search anchor for identifying musicians.
always: true
---

# Musician: Life Event Anchoring

## When to use
When a musician question describes distinctive personal experiences rather than discography clues.

## Technique
The most distinctive information about musicians is often unique life events, not musical works. Life events are far more unique and searchable than album/song names.

Priority ranking for biographical constraints:
1. **Cause of death / special events:** "died of professional negligence", "triple bypass surgery" — extremely rare, almost always a direct hit
2. **Personal life details:** "five children", "divorced twice", "searched for biological father at age 17"
3. **Education / career turning points:** "dropped out of college to pursue music"
4. **Work characteristics:** "first album songs written by Boris Vian" — use only when life events are unavailable

Also track auxiliary figures mentioned in questions: songwriters, doctors, family members.

## Query Templates
- `"musician [cause of death] charged sentenced [year]"`
- `"[musician name] "divorced" "children" wife"`
- `"singer dropped out [school type] pursue music [country]"`

## Worked Examples
**Dr. Kang Se Hoon:** Question mentioned hobbies of reading comics and playing computer games. Searched `"Shin Hae-chul solo album 1990s radio DJ hobbies comics computer games"` → then tracked the doctor: `"Shin Hae-chul Kang Se-hoon charged professional negligence death sentenced"`.

**Vanic:** Born in May, dropped out of school for music production. After narrowing candidates, verified with `"Vanic Jesse born May 1987 1988 1989 1990 Vancouver producer"`.

## Anti-pattern
Over-relying on album or song name searches. Album names are often not distinctive enough ("First Album", "The Light"). Anchor with life events first, then use discography only for verification.
