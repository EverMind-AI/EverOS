# EverMemOS Overview

[Home](../README.md) > [Docs](README.md) > Overview

## Core Vision

Build AI memory that never forgets, making every conversation built on previous understanding.

## What is EverMemOS?

**EverMemOS** is an open-source project designed to provide long-term memory capabilities to conversational AI agents. It extracts, structures, and retrieves information from conversations, enabling agents to maintain context, recall past interactions, and progressively build user profiles. This results in more personalized, coherent, and intelligent conversations.

**EverMemOS** is a forward-thinking **intelligent system**. While traditional AI memory serves merely as a "look-back" database, EverMemOS enables AI not only to "remember" what happened, but also to "understand" the meaning behind these memories and use them to guide current actions and decisions. In the EverMemOS demo tools, you can see how EverMemOS extracts important information from your history, and then remembers your preferences, habits, and history during conversations, just like a **friend** who truly knows you.

On the **LoCoMo** benchmark, our approach built upon EverMemOS achieved a reasoning accuracy of **92.3%** (evaluated by LLM-Judge), outperforming comparable methods in our evaluation.

> 💬 **More than memory — it's foresight.**

> 📄 **Paper Coming Soon** - Our technical paper is in preparation. Stay tuned!

---

## Unique Advantages

<table>
  <tr>
    <td width="33%" valign="top">
      <h3>🔗 Coherent Narrative</h3>
      <p><strong>Beyond "fragments," connecting "stories"</strong>: Automatically linking conversation pieces to build clear thematic context, enabling AI to "truly understand."</p>
      <blockquote>
        When facing multi-threaded conversations, it naturally distinguishes between "Project A progress discussion" and "Team B strategy planning," maintaining coherent contextual logic within each theme.<br/><br/>
        From scattered phrases to complete narratives, AI no longer just "understands one sentence" but "understands the whole story."
      </blockquote>
    </td>
    <td width="33%" valign="top">
      <h3>🧠 Evidence-Based Perception</h3>
      <p><strong>Beyond "retrieval," intelligent "perception"</strong>: Proactively capturing deep connections between memories and tasks, enabling AI to "think thoroughly" at critical moments.</p>
      <blockquote>
        Imagine: When a user asks for "food recommendations," the AI proactively recalls "you had dental surgery two days ago" as a key piece of information, automatically adjusting suggestions to avoid unsuitable options.<br/><br/>
        This is <strong>Contextual Awareness</strong> — enabling AI thinking to be truly built on understanding rather than isolated responses.
      </blockquote>
    </td>
    <td width="33%" valign="top">
      <h3>💾 Living Profiles</h3>
      <p><strong>Beyond "records," dynamic "growth"</strong>: Real-time user profile updates that get to know you better with each conversation, enabling AI to "recognize you authentically."</p>
      <blockquote>
        Every interaction subtly updates the AI's understanding of you — preferences, style, and focus points all continuously evolve.<br/><br/>
        As interactions deepen, it doesn't just "remember what you said," but is "learning who you are."
      </blockquote>
    </td>
  </tr>
</table>

---

## System Framework

EverMemOS operates along two main tracks: **memory construction** and **memory perception**. Together they form a cognitive loop that continuously absorbs, consolidates, and applies past information, so every response is grounded in real context and long-term memory.

<p align="center">
  <img src="../figs/overview.png" alt="Overview" />
</p>

### 🧩 Memory Construction

Memory construction layer: builds structured, retrievable long-term memory from raw conversation data.

- **Core elements**
  - ⚛️ **Atomic memory unit MemCell**: the core structured unit distilled from conversations for downstream organization and reference
  - 🗂️ **Multi-level memory**: integrate related fragments by theme and storyline to form reusable, hierarchical memories
  - 🏷️ **Multiple memory types**: covering episodes, profiles, preferences, relationships, semantic knowledge, basic facts, and core memories

- **Workflow**
  1. **MemCell extraction**: identify key information in conversations to generate atomic memories
  2. **Memory construction**: integrate by theme and participants to form episodes and profiles
  3. **Storage and indexing**: persist data and build keyword and semantic indexes to support fast recall

### 🔎 Memory Perception

Memory perception layer: quickly recalls relevant memories through multi-round reasoning and intelligent fusion, achieving precise contextual awareness.

#### 🎯 Intelligent Retrieval Tools

- **🧪 Hybrid Retrieval (RRF Fusion)**
  Parallel execution of semantic and keyword retrieval, seamlessly fused using Reciprocal Rank Fusion algorithm

- **📊 Intelligent Reranking (Reranker)**
  Batch concurrent processing with exponential backoff retry, maintaining stability under high throughput
  Reorders candidate memories by deep relevance, prioritizing the most critical information

#### 🚀 Flexible Retrieval Strategies

- **⚡ Lightweight Fast Mode**
  For latency-sensitive scenarios, skip LLM calls and use pure keyword retrieval (BM25)
  Achieve a faster response speed

- **🎓 Agentic Multi-Round Recall**
  For insufficient cases, generate 2-3 complementary queries, retrieve and fuse in parallel
  Enhance coverage of complex intents through multi-path RRF fusion

#### 🧠 Reasoning Fusion

- **Context Integration**: Concatenate recalled multi-level memories (episodes, profiles, preferences) with current conversation
- **Traceable Reasoning**: Model generates responses based on explicit memory evidence, avoiding hallucination

💡 Through the cognitive loop of **"Structured Memory → Multi-Strategy Recall → Intelligent Retrieval → Contextual Reasoning"**, the AI always "thinks with memory", achieving true contextual awareness.

---

## Why EverMemOS?

Traditional AI systems lack persistent memory, treating each conversation in isolation. EverMemOS changes this by:

1. **Extracting structured knowledge** from unstructured conversations
2. **Building coherent narratives** that connect related information
3. **Enabling intelligent perception** that goes beyond simple keyword matching
4. **Maintaining living profiles** that evolve with each interaction

This results in AI that doesn't just respond, but truly understands and remembers.

---

## Use Cases

EverMemOS is ideal for:

- **Personal AI Assistants** - Remember user preferences, habits, and history across sessions
- **Customer Service** - Maintain customer context and history for personalized support
- **Group Collaboration** - Track multi-participant conversations and team dynamics
- **Research & Analysis** - Build knowledge bases from conversation data
- **Educational Tools** - Adapt to student learning patterns and progress

---

## Next Steps

- **[Quick Setup](installation/SETUP.md)** - Get EverMemOS running
- **[Architecture](ARCHITECTURE.md)** - Deep dive into system design
- **[Usage Examples](usage/USAGE_EXAMPLES.md)** - Learn how to use EverMemOS
- **[API Documentation](api_docs/memory_api.md)** - Integrate with your application

---

## See Also

- [Architecture Design](ARCHITECTURE.md)
- [Memory Types Guide](dev_docs/memory_types_guide.md)
- [Development Guide](dev_docs/development_guide.md)
