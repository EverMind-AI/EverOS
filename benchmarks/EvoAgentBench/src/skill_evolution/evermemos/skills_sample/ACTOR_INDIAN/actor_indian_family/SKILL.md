---
name: actor_indian_family
description: >
  Track Indian actors through family chains — grandfather/father/spouse relationships are key indirect clues
always: true
---

# Family Chain Tracking for Indian Actors

## When to use
When the question points to the target Indian actor indirectly through family members (father, grandfather, spouse, in-laws) rather than naming the actor directly.

## Technique
The Indian film industry has dense family ties — many questions exploit this by describing a father's career, a spouse's achievements, or a grandfather's legacy to indirectly identify the target. You must track the family chain step by step rather than trying to jump directly to the final answer.

Workflow: (1) Identify the family member described most distinctively. (2) Search for that family member first. (3) Once identified, search "[family member name] son/daughter/spouse/grandchild" to find the target. (4) Verify the target against remaining question constraints.

## Query Templates
- `[actor name] father mother spouse family [relative's characteristics]`
- `[family member name] son daughter children actor Bollywood`
- `[actor name] spouse [spouse characteristics] family`

## Worked Examples

### Example
Question asks about a person whose father is an actor (born 1940-1960), father married an actress (1970-1990), and there's an interview biography book about the father.
- First identified the father through awards: `"actor award \"Dadasaheb Phalke\" born 1920 1930"`
- Then tracked to the child: `"Amitabh Bachchan" children son daughter actor`
- Verified biography book: `"Amitabh Bachchan" book "interviews" biography`
- Key: The father was the identifiable link; the target was reached through the family chain

## Anti-pattern
**Skipping the family chain**: Trying to search directly for the final target without first identifying the described family member. Questions using grandfather/father-in-law as clues require step-by-step tracking.
