# EverMemOS Configuration Guide

This guide provides a detailed explanation of the configuration options in `env.template`. Before deploying EverMemOS, please copy `env.template` to `.env` and fill in your actual configuration values according to this guide.

> **⚠️ Security Notice**:
> The `.env` file contains sensitive information (such as API keys and database passwords). Be sure to add it to `.gitignore` and **NEVER** commit it to version control systems.

---

## 1. LLM Configuration

Configuration for the LLM service used for memory extraction, Agentic retrieval, and Q&A generation.

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `LLM_PROVIDER` | Yes | LLM provider, usually set to `openai` for compatibility with OpenAI SDK format | `openai` |
| `LLM_MODEL` | Yes | Model name. **Evaluation** recommends `gpt-4o-mini`, **Demo** can use cost-effective models like `x-ai/grok-4-fast` | `gpt-4o-mini` |
| `LLM_BASE_URL` | Yes | API base URL, supports compatible interfaces like OpenRouter, DeepSeek | `https://openrouter.ai/api/v1` |
| `LLM_API_KEY` | Yes | Your API key | `sk-or-v1-xxxx` |
| `LLM_TEMPERATURE` | No | Generation temperature, lower values recommended for stable output | `0.3` |
| `LLM_MAX_TOKENS` | No | Maximum generation tokens | `32768` |

---

## 2. Vectorize Service Configuration

Configuration for converting text to vectors (Embeddings), supporting DeepInfra and vLLM.

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `VECTORIZE_PROVIDER` | Yes | Provider options: `deepinfra`, `vllm` | `deepinfra` |
| `VECTORIZE_API_KEY` | Yes* | API Key (Required for DeepInfra, Optional for vLLM) | `xxxxx` |
| `VECTORIZE_BASE_URL` | Yes | Service URL | `https://api.deepinfra.com/v1/openai` |
| `VECTORIZE_MODEL` | Yes | Model name, must match the server-side name | `Qwen/Qwen3-Embedding-4B` |
| `VECTORIZE_DIMENSIONS` | No | Vector dimensions. Set to `0` if vLLM doesn't support this parameter, otherwise keep model dimensions (e.g., `1024`) | `1024` |

**Advanced Settings**:
- `VECTORIZE_TIMEOUT`: Request timeout (seconds)
- `VECTORIZE_MAX_RETRIES`: Maximum retry attempts
- `VECTORIZE_BATCH_SIZE`: Batch size
- `VECTORIZE_MAX_CONCURRENT`: Maximum concurrent requests
- `VECTORIZE_ENCODING_FORMAT`: Encoding format, usually `float`

---

## 3. Rerank Service Configuration

Configuration for re-ranking retrieval results to improve relevance.

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `RERANK_PROVIDER` | Yes | Provider options: `deepinfra`, `vllm` | `deepinfra` |
| `RERANK_API_KEY` | Yes* | API Key | `xxxxx` |
| `RERANK_BASE_URL` | Yes | Service URL | `https://api.deepinfra.com/v1/inference` |
| `RERANK_MODEL` | Yes | Model name | `Qwen/Qwen3-Reranker-4B` |

**Advanced Settings**:
- `RERANK_TIMEOUT`: Timeout (seconds)
- `RERANK_BATCH_SIZE`: Batch size
- `RERANK_MAX_CONCURRENT`: Maximum concurrent requests

---

## 4. Database Configuration

EverMemOS relies on multiple database services, typically started via Docker Compose.

### Redis
Used for caching and distributed locks.
- `REDIS_HOST`: Host address (default `localhost`)
- `REDIS_PORT`: Port (default `6379`)
- `REDIS_DB`: Database index (default `8`)

### MongoDB
Primary database, stores memory cells, profiles, and conversation records.
- `MONGODB_HOST`: Host address (default `localhost`)
- `MONGODB_PORT`: Port (default `27017`)
- `MONGODB_USERNAME`: Username (default `admin`)
- `MONGODB_PASSWORD`: Password (default `memsys123`)
- `MONGODB_DATABASE`: Database name (default `memsys`)

### Elasticsearch
Used for keyword retrieval (BM25).
- `ES_HOSTS`: Service address (default `http://localhost:19200`)
- `SELF_ES_INDEX_NS`: Index namespace (default `memsys`)

### Milvus
Vector database, used for semantic retrieval.
- `MILVUS_HOST`: Host address (default `localhost`)
- `MILVUS_PORT`: Port (default `19530`)
- `SELF_MILVUS_COLLECTION_NS`: Collection namespace (default `memsys`)

---

## 5. Other Configuration

### API Server
- `API_BASE_URL`: Base URL for V1 API, used for client connections (default `http://localhost:1995`)

### Environment & Logging
- `LOG_LEVEL`: Log level (`INFO`, `DEBUG`, `WARNING`, `ERROR`)
- `ENV`: Environment identifier (`dev`, `prod`)
- `MEMORY_LANGUAGE`: Primary system language (`zh`, `en`)

---

## Configuration Examples

### 1. Using DeepInfra (Recommended)

```bash
VECTORIZE_PROVIDER=deepinfra
VECTORIZE_API_KEY=your_key_here
VECTORIZE_BASE_URL=https://api.deepinfra.com/v1/openai
VECTORIZE_MODEL=Qwen/Qwen3-Embedding-4B

RERANK_PROVIDER=deepinfra
RERANK_API_KEY=your_key_here
RERANK_BASE_URL=https://api.deepinfra.com/v1/inference
RERANK_MODEL=Qwen/Qwen3-Reranker-4B
```

### 2. Using Local vLLM

```bash
VECTORIZE_PROVIDER=vllm
VECTORIZE_API_KEY=none
VECTORIZE_BASE_URL=http://localhost:8000/v1
VECTORIZE_MODEL=Qwen3-Embedding-4B
VECTORIZE_DIMENSIONS=0  # vLLM sometimes requires disabling this parameter

RERANK_PROVIDER=vllm
RERANK_API_KEY=none
RERANK_BASE_URL=http://localhost:12000/score
RERANK_MODEL=Qwen3-Reranker-4B
```

> **ℹ️ vLLM Deployment Tips**:
> - **Embedding Models** (Supported since v0.4.0+):
>   ```bash
>   vllm serve Qwen/Qwen3-Embedding-4B --task embed --trust-remote-code
>   ```
> - **Reward/Reranker Models** (See [vLLM PR #19260](https://github.com/vllm-project/vllm/pull/19260) for details):
>   ```bash
>   vllm serve Qwen/Qwen3-Reranker-4B --task reward --trust-remote-code
>   ```
>   Note: Use `--task reward` for Reranker models.


