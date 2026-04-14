---
name: paper_term_translation
description: >
  Translate generalized descriptions in questions into precise academic terminology for effective paper search
always: true
---

# Terminology Translation for Research Papers

## When to use
When a question describes a research paper using colloquial or generalized language (e.g., "conjugal happiness", "sharp stones for harvest") instead of standard academic terms.

## Technique
Questions about research papers rarely use the exact terminology from the paper. The first step is always to translate vague descriptions into candidate academic terms. Think about what the actual paper title or abstract would say.

Try multiple synonym candidates. For example, "conjugal happiness" could be "marital satisfaction", "relationship quality", or "spousal well-being". Generate 2-3 plausible academic translations and search each.

## Query Templates
- `"[academic term]" [numerical value] [country/institution] [year]`
- `[methodology keyword] "[translated term]" [discipline] [year range]`

## Worked Examples

### Example
- Question: DOI of a paper exploring the prognosis of marital happiness, three authors, sample size of 150-400 couples
- Successful query: `"prognosis" "conjugal happiness" December 2023 three authors DOI`
- Why it worked: Directly used the key phrase "conjugal happiness" with quote marks for exact matching, combined with numerical constraints

## Anti-pattern
Searching only with the generalized description from the question (e.g., "prognosis conjugal happiness") without adding numerical constraints or year filters — results in too many irrelevant hits.
