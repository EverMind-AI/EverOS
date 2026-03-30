# Memory Retrieval Strategies

[Home](../../README.md) > [Docs](../README.md) > [Advanced](.) > Retrieval Strategies

This guide explains the different retrieval strategies available in EverMemOS and when to use each one.

---

## Table of Contents

- [Overview](#overview)
- [Lightweight Retrieval](#lightweight-retrieval)
- [Agentic Retrieval](#agentic-retrieval)
- [Choosing a Strategy](#choosing-a-strategy)
- [API Examples](#api-examples)
- [Performance Comparison](#performance-comparison)
- [Best Practices](#best-practices)

---

## Overview

EverMemOS provides two main retrieval strategies:

1. **Lightweight Retrieval** - Fast, efficient retrieval for latency-sensitive scenarios
2. **Agentic Retrieval** - Intelligent, multi-round retrieval for complex queries

Both strategies leverage the Memory Perception layer to recall relevant memories through multi-round reasoning and intelligent fusion, achieving precise contextual awareness.

---

## Lightweight Retrieval

Fast retrieval mode that skips LLM calls for minimum latency.

### Retrieval Modes

#### 1. Keyword Search

Pure keyword-based search using Elasticsearch BM25.

**Characteristics:**
- Fastest retrieval mode
- No embedding required
- Best for exact keyword matches
- Lower accuracy for semantic queries

**When to use:**
- Exact phrase or keyword search
- Latency is critical (< 100ms)
- No semantic understanding needed

**Example:**
```python
{
    "query": "soccer weekend",
    "retrieve_method": "keyword"
}
```

#### 2. Vector (Semantic Search)

Pure vector-based search using Milvus.

**Characteristics:**
- Semantic understanding
- Finds similar meaning, not just keywords
- Requires embedding model
- Moderate latency (~200-500ms)

**When to use:**
- Semantic similarity important
- Query phrasing differs from stored content
- Need conceptual matches

**Example:**
```python
{
    "query": "What sports does the user enjoy?",
    "retrieve_method": "vector"
}
```

#### 3. RRF (Hybrid Retrieval) - Recommended

Reciprocal Rank Fusion of BM25 and Embedding results.

**Characteristics:**
- Best of both worlds
- Parallel execution of BM25 and embedding search
- Fuses results using RRF algorithm
- Balanced accuracy and speed

**When to use:**
- Default choice for most scenarios
- Want both keyword and semantic matching
- Need robust retrieval across query types

**Example:**
```python
{
    "query": "What are the user's weekend activities?",
    "retrieve_method": "rrf"
}
```

### Intelligent Reranking

Optional reranking step to improve result relevance:

- **Batch concurrent processing** with exponential backoff retry
- **Deep relevance scoring** using reranker models
- **Prioritization** of most critical information
- **High throughput** stability

Reranking is automatically applied for `hybrid` and `agentic` retrieval methods. For programmatic control, see the [Agentic Retrieval Guide](../dev_docs/agentic_retrieval_guide.md).

---

## Agentic Retrieval

Intelligent, multi-round retrieval using LLM for query expansion and fusion.

### How It Works

1. **Query Analysis** - LLM analyzes the user query
2. **Query Expansion** - Generates 2-3 complementary queries
3. **Parallel Retrieval** - Retrieves memories for each query
4. **RRF Fusion** - Fuses results using multi-path RRF
5. **Context Integration** - Concatenates memories with current conversation

### Characteristics

- **Higher latency** (~2-5 seconds with LLM calls)
- **Better coverage** for complex intents
- **Multi-aspect queries** handled effectively
- **Adaptive** to query complexity

### When to Use

- Complex, multi-faceted queries
- Queries requiring context understanding
- When accuracy is more important than speed
- Insufficient results from lightweight modes

### Example Workflow

**User Query:** "Tell me about my work-life balance"

**Step 1 - Query Expansion:**
- Original: "Tell me about my work-life balance"
- Expanded 1: "work schedule and working hours"
- Expanded 2: "hobbies and leisure activities"
- Expanded 3: "stress and relaxation"

**Step 2 - Parallel Retrieval:**
Each query retrieves top-k memories using RRF

**Step 3 - Fusion:**
Results merged using multi-path RRF

**Step 4 - Response:**
LLM generates response based on retrieved memories

---

## Choosing a Strategy

### Decision Flow

```
Is latency critical (< 100ms)?
├─ Yes → Use Keyword
└─ No → Continue

Do you need semantic understanding?
├─ No → Use Keyword
└─ Yes → Continue

Is the query complex or multi-faceted?
├─ Yes → Use Agentic
└─ No → Continue

Default choice → Use RRF
```

### Use Case Matrix

| Use Case | Recommended Strategy | Reason |
|----------|---------------------|--------|
| Exact phrase search | Keyword | Fast, precise keyword matching |
| Product search by name | Keyword or RRF | Keywords important |
| Conversational queries | RRF or Agentic | Semantic understanding needed |
| Complex analysis questions | Agentic | Multi-aspect coverage |
| Real-time chat | RRF | Balance of speed and accuracy |
| Background indexing | Any | No latency constraints |
| Autocomplete/suggestions | Keyword | Speed critical |
| Research/analysis | Agentic | Accuracy critical |

---

## API Examples

### Lightweight - Keyword

```bash
curl -X GET http://localhost:1995/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "soccer",
    "user_id": "user_001",
    "memory_types": ["episodic_memory"],
    "retrieve_method": "keyword",
    "top_k": 5
  }'
```

### Lightweight - Vector

```bash
curl -X GET http://localhost:1995/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What sports does the user like?",
    "user_id": "user_001",
    "memory_types": ["episodic_memory"],
    "retrieve_method": "vector",
    "top_k": 5
  }'
```

### Lightweight - RRF (Recommended)

```bash
curl -X GET http://localhost:1995/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Tell me about the user hobbies",
    "user_id": "user_001",
    "memory_types": ["episodic_memory"],
    "retrieve_method": "rrf",
    "top_k": 5
  }'
```

### Agentic Retrieval

```bash
curl -X GET http://localhost:1995/api/v1/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is my work-life balance like?",
    "user_id": "user_001",
    "memory_types": ["episodic_memory"],
    "retrieve_method": "agentic",
    "top_k": 10
  }'
```

---

## Performance Comparison

### Latency

| Strategy | Typical Latency | Notes |
|----------|----------------|-------|
| Keyword | 50-100ms | Fastest |
| Vector | 200-500ms | Depends on Milvus performance |
| RRF | 200-600ms | Parallel keyword + vector |
| Agentic | 2-5 seconds | Includes LLM query expansion |

### Accuracy

Measured on LoCoMo benchmark:

| Strategy | Precision | Recall | F1 Score |
|----------|-----------|--------|----------|
| Keyword | 0.72 | 0.68 | 0.70 |
| Vector | 0.78 | 0.75 | 0.77 |
| RRF | 0.85 | 0.82 | 0.84 |
| Agentic | 0.91 | 0.89 | 0.90 |

*Note: Actual performance varies by query type and data*

### Resource Usage

| Strategy | CPU | Memory | Network |
|----------|-----|--------|---------|
| Keyword | Low | Low | Minimal |
| Vector | Medium | Medium | Moderate (embedding API) |
| RRF | Medium | Medium | Moderate |
| Agentic | Medium-High | Medium | High (multiple LLM calls) |

---

## Best Practices

### 1. Start with RRF

For most applications, RRF provides the best balance:
- Good accuracy
- Reasonable latency
- Robust across query types

### 2. Use Keyword Search for Known Patterns

When users search for specific keywords or phrases:
- Product names
- Exact quotes
- Technical terms

### 3. Reserve Agentic for Complex Queries

Use agentic retrieval when:
- User query is vague or complex
- Standard retrieval returns insufficient results
- Analysis or reasoning required

### 4. Tune top_k Parameter

- **Keyword**: Lower top_k (3-5) for precise matches
- **Vector/RRF**: Medium top_k (5-10) for coverage
- **Agentic**: Higher top_k (10-20) for comprehensive results

### 5. Monitor and Optimize

- Track query latency and adjust strategy
- Monitor result relevance and switch modes
- Consider caching frequent queries

### 6. Combine Strategies

Use different strategies for different query types:

```python
def select_strategy(query):
    # Exact phrase (in quotes)
    if query.startswith('"') and query.endswith('"'):
        return "keyword"

    # Complex question
    if any(word in query.lower() for word in ["why", "how", "explain", "analyze"]):
        return "agentic"

    # Default
    return "rrf"
```

---

## See Also

- [Architecture: Memory Perception](../ARCHITECTURE.md#memory-perception-architecture) - Technical architecture
- [API Documentation](../api_docs/memory_api.md) - Complete API reference
- [Agentic Retrieval Guide](../dev_docs/agentic_retrieval_guide.md) - In-depth agentic retrieval
- [Evaluation Guide](../../evaluation/README.md) - Benchmarking retrieval strategies
- [Usage Examples](../usage/USAGE_EXAMPLES.md) - Practical examples
