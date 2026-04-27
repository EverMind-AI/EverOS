---
description: Ask a question about past work. Searches memories and combines with current context to answer.
arguments:
  - name: question
    description: The question to answer
    required: true
---

# EverMem Ask

Answer a question using **both** memory search results **and** current conversation context.

## Question
{{question}}

## Instructions

1. **Search memories** using `evermem_search` MCP tool with relevant keywords. Start with 10 results.

2. **Evaluate results**:
   - If memories provide useful context, note what you learned
   - If more detail needed, search again with different keywords (up to 3 searches)
   - If no relevant memories found, that's OK - proceed with what you know

3. **Combine sources** to answer:
   - Memory search results (past sessions)
   - Current conversation context (this session)
   - Your general knowledge (when applicable)

4. **Be honest about sources**:
   - "Based on our discussion on [date]..." - when citing memory
   - "From our current session..." - when citing current context
   - "I don't have any recorded information about this" - when memories don't help
   - "Based on general best practices..." - when using general knowledge

5. **Admit uncertainty**:
   - If memories are incomplete or unclear, say so
   - If you're inferring rather than recalling, make that clear
   - It's better to say "I don't know" than to guess

## Response Format

Start with a direct answer, then provide supporting context:

```
[Direct answer to the question]

**From memories:**
- [Relevant points from past sessions, with dates]

**Current context:**
- [Relevant points from this session, if any]

**Note:** [Any caveats or gaps in knowledge]
```

Now answer the user's question.
