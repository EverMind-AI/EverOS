---
name: paper_author_tracking
description: >
  Once an author name is discovered from initial search results, use the author name for precise follow-up searches
always: true
---

# Author Tracking for Research Papers

## When to use
When initial searches return candidate papers or author names, and you need to confirm the correct paper or find additional details about the author.

## Technique
After the first round of keyword search, extract author names from the results. Then immediately conduct a precise tracking search using "author full name + keywords". Author names are highly specific identifiers — combining them with topic or institution constraints almost always locates the exact paper.

This is a two-phase strategy: Phase 1 discovers the author name through broad search; Phase 2 uses the author name as an anchor for precise search.

## Query Templates
- `"[author full name]" [journal/institution] [year] [topic keyword]`
- `"[author full name]" [birth date / biographical detail]`

## Worked Examples

### Example
- Question: Article published between 1988-1998, tracking historical processes through changes in the political landscape
- Successful query: `Robert D. Kaplan birth date born`
- Why it worked: First located the author Robert D. Kaplan through topic search, then used the author name to answer the specific biographical question

## Anti-pattern
After discovering an author name in the first round, failing to use it for continued searching and instead repeatedly adjusting keyword combinations — wastes search rounds.
