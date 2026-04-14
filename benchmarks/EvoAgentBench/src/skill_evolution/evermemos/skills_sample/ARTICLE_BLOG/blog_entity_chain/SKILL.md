---
name: blog_entity_chain
description: >
  Track from one entity to a related entity through multi-step searches — author to article to mentioned person/place
always: true
---

# Entity Chain Tracking for Blogs

## When to use
When a question involves a chain of related entities — e.g., an article's author mentions a person who appeared in a film, or a theater manager who also acted in a movie.

## Technique
Blog and article questions often require following a chain: Author -> Article -> Mentioned Person -> Related Fact. Each link requires a separate search. After finding one entity, immediately use it as an anchor to search for the next entity in the chain.

Do not try to find the final answer in one search. Break the chain into steps and track each link.

## Query Templates
- `"[person name]" [profession/institution] [year] [keywords]`
- `"[entity found in step 1]" [relationship keyword] [next entity clue]`

## Worked Examples

### Example
- Question: A historical article about a theater that opened in 1930 (published in 2017), where the theater manager also appeared in a 2008 horror film
- Successful query: `"Dead on Site" 2008 Tamara Mack theatre house manager`
- Why it worked: First found the theater name and manager name through theater history search, then cross-searched using manager name + film genre to complete the entity chain

## Anti-pattern
Not tracking intermediate entities — after finding a person's name, not using it for continued searching and instead repeating generic topic keyword searches.
