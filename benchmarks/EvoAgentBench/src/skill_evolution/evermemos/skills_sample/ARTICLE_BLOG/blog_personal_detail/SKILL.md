---
name: blog_personal_detail
description: >
  Use unique personal details from blog posts (daily habits, experiences, quirky facts) as highly distinctive search keywords
always: true
---

# Personal Detail Search for Blogs

## When to use
When a question describes a blog or personal article that mentions unique personal habits, routines, or life experiences of the author.

## Technique
Blogs are filled with unique personal details — daily habits, work experiences, pet names, food preferences, city names. These are far more distinctive than generic topic keywords. A phrase like "eating the same breakfast every day" or "watching cartoons after 9pm" is nearly unique across the entire web.

Extract the most unusual personal details from the question and use them as quoted search terms. Combine 2-3 such details for maximum precision.

## Query Templates
- `"[unique habit/detail]" "[another unique detail]" [author profession] [context]`
- `blog [year] "[personal experience phrase]" [topic]`

## Worked Examples

### Example
- Question: A software developer simultaneously attending university, 2021 article mentioning full-time work, eating the same breakfast every day, watching cartoons
- Successful query: `"same breakfast" "watching cartoons" software developer routine`
- Why it worked: Used unique personal habit descriptions as quoted exact searches — these details are nearly unique across the entire corpus

## Anti-pattern
Over-reliance on generic topic search (e.g., "software developer blog") without adding personal details — blog topics are rarely distinctive enough on their own.
