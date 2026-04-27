---
description: Guidance for using EverMem memory tools to recall past session context
alwaysInclude: true
---

# EverMem Memory Tools

You have access to memory tools that can recall context from the user's past coding sessions. Use these tools proactively when they would help provide better assistance.

## Available Tools

- **search_memories**: Search past conversations using semantic + keyword matching
- **get_memory**: Retrieve full details of a specific memory by ID

## When to Use Memory Search

**DO search memories when:**
- User asks about past work, decisions, or implementations ("how did we handle X?")
- User references previous sessions ("remember when", "last time", "we discussed")
- User is debugging something that may have been solved before
- User asks about project patterns, conventions, or architecture decisions
- Context from previous sessions would improve your response
- User seems to expect you to know something from before

**DON'T search memories when:**
- The question is self-contained and doesn't need historical context
- User explicitly provides all needed context in their message
- It's a general knowledge question unrelated to their project history
- You've already searched for this topic in the current session

## Best Practices

1. **Be selective**: Don't search for every query - only when past context adds value
2. **Use specific queries**: Search for relevant terms, not the entire user message
3. **Synthesize results**: When you find relevant memories, integrate them naturally into your response
4. **Be transparent**: Mention when your response is informed by past session context
