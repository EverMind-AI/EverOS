# Agentic Retrieval Testing Guide

## Overview

This document explains how to test the Agentic retrieval feature of the V1 API. Agentic retrieval is an LLM-guided intelligent multi-round retrieval method that can automatically determine the sufficiency of retrieval results and perform multi-round optimization.

## Features

### Agentic Retrieval Process

1. **Round 1**: RRF hybrid retrieval (Embedding + BM25)
2. **Rerank**: Use Reranker to optimize result quality
3. **LLM Judgment**: Use LLM to determine if results are sufficient
4. **Round 2** (if needed): 
   - LLM generates multiple refined queries
   - Parallel retrieval for all queries
   - Fusion and Rerank to return final results

### API Endpoint

```
POST /api/v0/memories/search
```

### Request Format

```json
{
  "query": "What does the user like to eat?",
  "user_id": "default",
  "group_id": "assistant",
  "time_range_days": 365,
  "top_k": 20,
  "llm_config": {
    "api_key": "your_api_key",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini"
  }
}
```

### Response Format

```json
{
  "status": "ok",
  "message": "Agentic retrieval successful, found 15 memories",
  "result": {
    "memories": [...],
    "count": 15,
    "metadata": {
      "retrieval_mode": "agentic",
      "is_multi_round": true,
      "round1_count": 20,
      "is_sufficient": false,
      "reasoning": "Need more specific information about dietary preferences",
      "refined_queries": ["What is the user's favorite cuisine?", "What does the user dislike eating?"],
      "round2_count": 40,
      "final_count": 15,
      "total_latency_ms": 2345.67
    }
  }
}
```

## Testing Instructions

### Running Tests

```bash
# Start the service
uv run python src/run.py 

# Run tests (in another terminal)
uv run python src/bootstrap.py demo/test_v3_retrieve_http.py
```

### Environment Configuration

Agentic retrieval requires LLM API Key configuration:

```bash
# Add to .env file
OPENROUTER_API_KEY=your_api_key
# or
OPENAI_API_KEY=your_api_key
```

If no API Key is configured, the test will automatically skip the Agentic retrieval part.

### Test Cases

The test file includes the following Agentic retrieval test cases:

1. **Simple Query**: "Beijing travel" - Test single-round retrieval (possibly sufficient)
2. **Complex Query**: "What does the user like to eat? What are their usual eating habits?" - Test multi-round retrieval
3. **Multi-dimensional Query**: "User's personality traits and hobbies" - Test multi-dimensional retrieval

### Expected Results

- **Single-round Retrieval**: If Round 1 results are sufficient, return directly
- **Multi-round Retrieval**: If Round 1 results are insufficient, LLM generates refined queries and proceeds to Round 2

## Performance Notes

- **Latency**: Typically 2-5 seconds (including LLM calls)
- **Cost**: Incurs LLM API call costs (approximately 2-3 calls)
- **Accuracy**: More accurate than regular retrieval, especially suitable for complex queries

## Integration with Chat Module

The chat module (`demo/chat_with_memory.py`) has integrated Agentic retrieval:

1. Select "Agentic Retrieval" when starting the chat application
2. The system will automatically use LLM-guided multi-round retrieval
3. Each conversation outputs detailed retrieval metadata

## Troubleshooting

### Issue 1: API Key Error

**Symptom**: Prompt "Missing LLM API Key"

**Solution**:
```bash
# Add to .env file
OPENROUTER_API_KEY=your_key_here
```

### Issue 2: Timeout

**Symptom**: Request timeout (over 60 seconds)

**Cause**: Agentic retrieval involves multiple LLM calls, which may timeout with slow network or LLM response

**Solution**:
- Check network connection
- Use a faster LLM model (such as gpt-4o-mini)
- Increase client timeout duration

### Issue 3: Empty Retrieval Results

**Symptom**: Returns 0 memories

**Cause**: No relevant data in database

**Solution**:
```bash
# Run data import first
uv run python src/bootstrap.py demo/extract_memory.py

# Then test retrieval
uv run python src/bootstrap.py demo/test_v3_retrieve_http.py
```

## References

- [Memory API Documentation](../api_docs/memory_api.md)
- [Agentic Retrieval Guide](./agentic_retrieval_guide.md)
- [Memory Manager Usage Guide](./api_usage_guide.md)

