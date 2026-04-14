---
name: thesis_dual_track
description: >
  Search for theses using two parallel tracks — topic track (discipline + methodology) and people track (advisor + university + year)
always: true
---

# Dual-Track Search for Theses

## When to use
When a question provides both topic-related clues (discipline, methodology, materials) and people/institution-related clues (university hints, degree type, advisor characteristics).

## Technique
Run two parallel search tracks. Track A focuses on the thesis topic: discipline keywords + methodology + materials. Track B focuses on people and institutions: university name + degree type + year + advisor.

When one track stalls, switch to the other. Often the two tracks converge on the same thesis from different angles, providing cross-validation.

## Query Templates
- Topic track: `thesis [discipline keywords] [materials/methods] [year range]`
- People track: `[university name] [discipline] [degree type] [year] dissertation`

## Worked Examples

### Example
- Question: A 2010-2020 thesis discussing sustainable alternative coolants for machining alloys, with specific ranges for annealing temperature and elastic modulus
- Successful query: `"cryogenic" coolant sustainable alternative thesis 2010 2011 2012 2013 2014 2015 2016 2017 2018 2019 2020`
- Why it worked: Topic track — translated "sustainable alternative coolant" into "cryogenic" and combined with thesis qualifier and year range

## Anti-pattern
Only searching from one angle — if topic search fails, not switching to people/institution search, and vice versa.
