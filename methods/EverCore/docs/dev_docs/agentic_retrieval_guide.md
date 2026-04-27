# Agentic Retrieval Guide

## Overview

Agentic retrieval is an LLM-guided multi-round retrieval method that significantly improves retrieval quality for complex queries through intelligent judgment and query optimization.

## Core Features

✅ **Intelligent Judgment**: LLM automatically determines if retrieval results are sufficient  
✅ **Multi-Round Retrieval**: Automatically performs a second round of retrieval when insufficient  
✅ **Multi-Query Strategy**: Generates 2-3 complementary queries to improve recall  
✅ **Automatic Fallback**: Falls back to Lightweight retrieval on failure  
✅ **Complete Metadata**: Returns detailed retrieval process information  

## Quick Start

### 1. Using in Chat Interface

Run `chat_with_memory.py` and select retrieval mode:

```bash
uv run python src/bootstrap.py demo/chat_with_memory.py
```

Select the 4th option: `Agentic Retrieval - LLM-guided multi-round retrieval (experimental)`

### 2. Using in Code

```python
from agentic_layer.memory_manager import MemoryManager
from memory_layer.llm.llm_provider import LLMProvider
from agentic_layer.agentic_utils import AgenticConfig

# Initialize LLM Provider
llm = LLMProvider(
    provider_type="openai",
    model="gpt-4",
    api_key="your_api_key",
    base_url="https://api.openai.com/v1",
    temperature=0.0,
)

# Initialize Memory Manager
memory_manager = MemoryManager()

# Execute Agentic retrieval
result = await memory_manager.retrieve_agentic(
    query="What foods does the user like?",
    group_id="food_lovers_group",
    llm_provider=llm,
    top_k=20,
)

# View results
print(f"Retrieved {result['count']} memories")
print(f"Is sufficient: {result['metadata']['is_sufficient']}")

if result['metadata']['is_multi_round']:
    print(f"Refined queries: {result['metadata']['refined_queries']}")
```

## Advanced Configuration

### Custom Agentic Configuration

```python
from agentic_layer.agentic_utils import AgenticConfig

# Create custom configuration
config = AgenticConfig(
    # Round 1 configuration
    round1_emb_top_n=50,        # Embedding candidates
    round1_bm25_top_n=50,       # BM25 candidates
    round1_top_n=20,            # Top N after RRF fusion
    round1_rerank_top_n=10,      # Top N after rerank for LLM judgment
    
    # LLM configuration
    llm_temperature=0.0,        # Low temperature for judgment
    llm_max_tokens=500,
    
    # Round 2 configuration
    enable_multi_query=True,    # Enable multi-query
    num_queries=3,              # Expected number of queries
    round2_per_query_top_n=50,  # Recall per query
    
    # Fusion configuration
    combined_total=40,          # Total after merging
    final_top_n=20,             # Final top N
    
    # Rerank configuration
    use_reranker=True,
    reranker_instruction="Rank based on relevance between query and memory",
)

# Use custom configuration
result = await memory_manager.retrieve_agentic(
    query="What foods does the user like?",
    group_id="food_lovers_group",
    llm_provider=llm,
    agentic_config=config,
)
```

## Return Format

```python
{
    "memories": [
        {
            "event_id": "...",
            "user_id": "...",
            "group_id": "...",
            "timestamp": "2024-01-15T10:30:00",
            "episode": "User said he loves Sichuan cuisine, especially Mapo Tofu",
            "summary": "User's cuisine preferences",
            "subject": "Eating habits",
            "score": 0.95
        },
        # ... more memories
    ],
    "count": 20,
    "metadata": {
        # Basic information
        "retrieval_mode": "agentic",
        "is_multi_round": True,  # Whether multi-round retrieval was performed
        
        # Round 1 statistics
        "round1_count": 20,
        "round1_reranked_count": 10,
        "round1_latency_ms": 800,
        
        # LLM judgment
        "is_sufficient": False,
        "reasoning": "Missing user's specific cuisine preferences and taste information",
        "missing_info": ["Cuisine preferences", "Taste habits", "Dietary restrictions"],
        
        # Round 2 statistics (only when multi-round)
        "refined_queries": [
            "What is the user's favorite cuisine?",
            "What flavors does the user like?",
            "What dietary restrictions does the user have?"
        ],
        "query_strategy": "Break down original query into multiple specific sub-questions",
        "num_queries": 3,
        "round2_count": 40,
        "round2_latency_ms": 600,
        "multi_query_total_docs": 120,
        
        # Final statistics
        "final_count": 20,
        "total_latency_ms": 3500
    }
}
```

## Workflow

```
User Query
  ↓
Round 1: Hybrid Search (Embedding + BM25 + RRF)
  ↓
RRF Fusion → Top 20
  ↓
Rerank → Top 10
  ↓
LLM Judges Sufficiency
  ↓
├─ Sufficient → Return Round 1's Top 20 ✅
│
└─ Insufficient → LLM generates multi-queries (2-3)
              ↓
          Round 2: Parallel retrieval for all queries
              ↓
          Multi-query RRF fusion
              ↓
          Deduplicate + merge to 40
              ↓
          Rerank → Top 20 ✅
```

## Performance Metrics

| Metric | Single Round (Sufficient) | Multi-Round (Insufficient) |
|--------|--------------------------|---------------------------|
| Latency | 2-5s | 5-10s |
| LLM Calls | 1 | 2 |
| Token Usage | ~500 | ~1500 |
| API Cost | ~$0.001 | ~$0.003 |

*Estimated values based on GPT-4*

## Use Cases

### ✅ Suitable for Agentic Retrieval

1. **Complex Queries**: Requires information from multiple perspectives
   - ❌ "What does the user like to eat?" (too broad)
   - ✅ "What is the user's favorite Sichuan dish and taste preferences?"

2. **Scattered Information**: Related memories distributed across different time points

3. **High Quality Requirements**: Scenarios requiring high recall and precision

### ❌ Not Suitable for Agentic Retrieval

1. **Simple Queries**: Questions that can be directly answered
   - "What day is it today?"
   - "What is the user's name?"

2. **Latency Sensitive**: Scenarios requiring < 1 second response

3. **Cost Sensitive**: Cannot afford LLM API costs

## Fallback Strategy

Agentic retrieval automatically falls back to Lightweight retrieval in the following cases:

1. ❌ LLM API call failure
2. ❌ Timeout (default 60 seconds)
3. ❌ `llm_provider` not provided
4. ❌ Candidate memories are empty

Fallback is marked in metadata:

```python
{
    "metadata": {
        "retrieval_mode": "agentic_fallback",
        "fallback_reason": "LLM API timeout"
    }
}
```

## Cost Optimization

### 1. Adjust LLM Model

```python
# Use cheaper model
llm = LLMProvider(
    provider_type="openai",
    model="gpt-4o-mini",  # Cheaper
    # model="gpt-4",      # More accurate but more expensive
)
```

### 2. Disable Multi-Query

```python
config = AgenticConfig(
    enable_multi_query=False,  # Only generate 1 query (reduce cost)
)
```

### 3. Disable Reranker

```python
config = AgenticConfig(
    use_reranker=False,  # Don't use reranker (reduce latency and cost)
)
```

## Troubleshooting

### Issue: LLM API Call Failure

**Reasons**:
- Incorrect API Key
- Network issues
- API rate limiting

**Solutions**:
1. Check API Key in `.env` file
2. Verify network connection
3. Check detailed error information in logs

### Issue: High Latency (> 10s)

**Reasons**:
- Slow LLM response
- Too many candidate memories
- Reranker timeout

**Solutions**:
1. Reduce `time_range_days` (reduce candidates)
2. Disable reranker
3. Use faster LLM model

### Issue: Poor Retrieval Quality

**Reasons**:
- Inaccurate LLM judgment
- Unreasonable query generation
- Prompt not adapted

**Solutions**:
1. Use stronger LLM model (e.g., GPT-4)
2. Adjust prompt template (in `agentic_utils.py`)
3. Increase `round1_rerank_top_n` (give LLM more samples)

## Comparison with Other Retrieval Modes

| Feature | Lightweight | Agentic |
|---------|------------|---------|
| Latency | 0.5-2s | 5-10s |
| LLM Calls | ❌ None | ✅ 1-2 |
| Multi-Round | ❌ No | ✅ Yes |
| Recall | Medium | High |
| Precision | Medium | High |
| Cost | Low | Medium |
| Use Cases | Simple queries | Complex queries |

## Best Practices

1. ✅ **Prioritize Lightweight**: For simple queries, Lightweight is sufficient
2. ✅ **Use Agentic for Complex Queries**: Only when needed
3. ✅ **Monitor Costs**: Track LLM token consumption
4. ✅ **Log Analysis**: Regularly review if LLM judgments are reasonable
5. ✅ **A/B Testing**: Compare effects of different modes

## Example: Complete Chat Flow

```python
import asyncio
from agentic_layer.memory_manager import MemoryManager
from memory_layer.llm.llm_provider import LLMProvider

async def main():
    # Initialize
    llm = LLMProvider("openai", model="gpt-4", api_key="...")
    memory_manager = MemoryManager()
    
    # User query
    query = "What foods does the user like? Any dietary restrictions?"
    
    # Execute retrieval
    result = await memory_manager.retrieve_agentic(
        query=query,
        group_id="food_lovers_group",
        llm_provider=llm,
    )
    
    # Display results
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}\n")
    
    print(f"Retrieval mode: {result['metadata']['retrieval_mode']}")
    print(f"Retrieved {result['count']} memories")
    print(f"Total latency: {result['metadata']['total_latency_ms']:.0f}ms\n")
    
    # LLM judgment
    print(f"LLM judgment: {'✅ Sufficient' if result['metadata']['is_sufficient'] else '❌ Insufficient'}")
    print(f"Reasoning: {result['metadata']['reasoning']}\n")
    
    # Multi-round information
    if result['metadata']['is_multi_round']:
        print(f"📝 Entered Round 2")
        print(f"Generated queries:")
        for i, q in enumerate(result['metadata']['refined_queries'], 1):
            print(f"  {i}. {q}")
        print()
    
    # Display memories
    print(f"Top 5 memories:")
    for i, mem in enumerate(result['memories'][:5], 1):
        print(f"\n[{i}] {mem['timestamp'][:10]}")
        print(f"    {mem['episode'][:100]}...")
        print(f"    Score: {mem['score']:.3f}")

if __name__ == "__main__":
    asyncio.run(main())
```

## More Resources

- 📖 [Memory Manager API Documentation](../docs/api_docs/memory_api.md)
- 🔬 [Retrieval Evaluation](../../evaluation/locomo_evaluation/README.md)
- 💡 [Best Practices](../docs/dev_docs/getting_started.md)

---

**Notes**:
- Agentic retrieval is an experimental feature and may be adjusted in future versions
- Please understand the costs and limitations of LLM APIs before using
- It is recommended to conduct thorough testing before deploying in production environments

