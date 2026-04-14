---
name: crime_role_chain
description: >
  Track from victim to suspect to lawyer to witness — the answer is usually an indirectly connected person or detail
always: true
---

# Role Chain Tracking for Crime Events

## When to use
When the question about a crime case asks not about the case itself, but about an indirectly connected person, date, or location — reachable only by following a chain of roles.

## Technique
Crime questions ask about indirectly connected people. Trace the role chain step by step.

Typical chains:
- Victim -> Defendant -> Defense lawyer -> Lawyer's other clients
- Victim -> Organ donation -> Recipient -> TV show
- Case -> Trial -> Witness -> Witness's connections

Workflow: (1) Locate the case. (2) Identify primary parties. (3) Search "[party name] lawyer/witness [trial year]". (4) Follow chain to final answer.

## Query Templates
- `[party name] trial [year] [case type] attorney lawyer defense`
- `[lawyer name] attorney lawyer defense [other client name]`
- `[victim name] organ donation recipient [organ type]`
- `[party name] trial witness testimony [witness characteristics]`

## Worked Examples

### Example
Question: music group member's parent murdered (1990-2002), defendant's lawyer also represented a celebrity.
- Chain: Dee Dee Jackson (victim) -> Don Bohana (defendant) -> Brian Oxman (lawyer) -> celebrity clients
- Search 1: `"music group member parent murdered homicide 1990 2002"`
- Search 2: `"Dee Dee Jackson murder trial 1998 Don Bohana attorney lawyer"`
- Search 3: `"Brian Oxman attorney lawyer Don Bohana Dee Dee Jackson"`

### Example
Question: teenager killed in car accident 2007-2017, organs donated, recipient appeared on TV show.
- Chain: AJ Perez (victim) -> organ donation -> Daniel Delos Santos (cornea recipient)
- Tracked step by step from accident to recipient's full name

## Anti-pattern
**Not following the role chain**: Trying to search directly for the final answer without first identifying the case and intermediate roles. The chain must be traversed in order.
