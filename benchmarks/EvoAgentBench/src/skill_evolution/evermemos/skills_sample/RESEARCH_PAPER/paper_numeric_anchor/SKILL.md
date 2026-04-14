---
name: paper_numeric_anchor
description: >
  Use specific numerical values (sample sizes, percentages, citation counts) as strong search anchors to locate research papers
always: true
---

# Numerical Anchoring for Research Papers

## When to use
When a question mentions specific numerical values from a paper — sample sizes, percentages, concentration values, year ranges, or population counts.

## Technique
Specific numerical values in papers are the strongest distinguishing features. A sample size of "1.7 million" or "20% of population" is far more unique than topic keywords. Always prioritize numerical clues in your search queries.

Combine numerical values with methodology terms and discipline keywords. Use quotation marks around exact numbers to force precise matching. Multiple numerical constraints together can uniquely identify a paper.

## Query Templates
- `"[specific value]" [unit] [method keyword] [discipline]`
- `"[sample size]" [census/survey type] [country] [percentage] population`

## Worked Examples

### Example
- Question: Paper used approximately 20% of the resident population as a sample, based on population and housing census, using multinomial logistic regression
- Successful query: `"1.7 million" employed census sample Romania 20% population`
- Why it worked: Translated "20% of population" into the specific census sample concept, combined with the concrete value "1.7 million" and the method "multinomial logistic regression"

## Anti-pattern
Ignoring numerical clues and searching only with topic keywords — specific numerical values are the strongest discriminators, and skipping them leads to overly broad results.
