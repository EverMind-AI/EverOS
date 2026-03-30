# Complete Setup Guide

[Home](../../README.md) > [Docs](../README.md) > [Installation](.) > Setup

This guide provides comprehensive instructions for installing and setting up EverMemOS.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Docker Installation (Recommended)](#docker-installation-recommended)
- [Environment Configuration](#environment-configuration)
- [Starting the Server](#starting-the-server)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)
- [Next Steps](#next-steps)

---

## System Requirements

### Minimum Requirements

- **Python**: 3.10 or higher
- **uv**: Package manager (will be installed during setup)
- **Docker**: 20.10+
- **Docker Compose**: 2.0+
- **RAM**: At least 4GB available (for Elasticsearch and Milvus)
- **Disk Space**: At least 10GB free

### Recommended Requirements

- **RAM**: 8GB or more
- **CPU**: 4 cores or more
- **Disk Space**: 20GB or more (especially for large datasets)

### Operating Systems

EverMemOS has been tested on:
- macOS (Intel and Apple Silicon)
- Linux (Ubuntu 20.04+, Debian, etc.)
- Windows (via WSL2)

---

## Installation Methods

EverMemOS can be installed in two ways:

1. **Docker Installation (Recommended)** - Use Docker Compose for all dependency services
2. **Manual Installation** - Install and configure each service manually

This guide covers the Docker installation method. For manual installation, see [Advanced Installation](#manual-installation-advanced).

---

## Docker Installation (Recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/EverMind-AI/EverMemOS.git
cd EverMemOS
```

### Step 2: Start Docker Services

Start all dependency services (MongoDB, Elasticsearch, Milvus, Redis) with one command:

```bash
docker-compose up -d
```

This will start:
- MongoDB on port 27017
- Elasticsearch on port 19200
- Milvus on port 19530
- Redis on port 6379

See [Docker Setup Guide](DOCKER_SETUP.md) for detailed service configuration.

### Step 3: Verify Docker Services

Check that all services are running:

```bash
docker-compose ps
```

You should see all services in the "Up" state.

### Step 4: Install uv Package Manager

If you don't have uv installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

After installation, restart your terminal or run:

```bash
source $HOME/.cargo/env
```

Verify installation:

```bash
uv --version
```

### Step 5: Install Project Dependencies

```bash
uv sync
```

This will:
- Create a virtual environment
- Install all required Python packages
- Set up the project for development

---

## Environment Configuration

### Step 1: Copy Environment Template

```bash
cp env.template .env
```

### Step 2: Configure API Keys

Edit the `.env` file and fill in the required configurations:

```bash
# Open .env in your preferred editor
nano .env
# or
vim .env
# or
code .env
```

### Required Configuration

#### LLM API Key (for memory extraction)

Choose one of the following:

```bash
# Option 1: OpenAI
LLM_API_KEY=sk-your-openai-key-here
LLM_API_BASE=https://api.openai.com/v1

# Option 2: OpenRouter
OPENROUTER_API_KEY=sk-or-v1-your-openrouter-key
OPENROUTER_API_BASE=https://openrouter.ai/api/v1

# Option 3: Other OpenAI-compatible API
LLM_API_KEY=your-api-key
LLM_API_BASE=https://your-api-endpoint.com/v1
```

#### Vectorize API Key (for embedding and reranking)

```bash
# DeepInfra (recommended)
VECTORIZE_API_KEY=your-deepinfra-key
VECTORIZE_API_BASE=https://api.deepinfra.com/v1/openai

# Or configure embedding and rerank separately
EMBEDDING_API_KEY=your-embedding-key
EMBEDDING_API_BASE=https://your-embedding-endpoint.com
RERANK_API_KEY=your-rerank-key
RERANK_API_BASE=https://your-rerank-endpoint.com
```

### Optional Configuration

```bash
# Model selection
LLM_MODEL=gpt-4  # or gpt-3.5-turbo, etc.
EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
RERANK_MODEL=BAAI/bge-reranker-large

# Service endpoints (default values shown)
MONGODB_URI=mongodb://admin:memsys123@localhost:27017
ELASTICSEARCH_URL=http://localhost:19200
MILVUS_HOST=localhost
MILVUS_PORT=19530
REDIS_URL=redis://localhost:6379
```

For complete configuration options, see the [Configuration Guide](../usage/CONFIGURATION_GUIDE.md).

---

## Starting the Server

### Start the API Server

```bash
uv run python src/run.py --port 1995
```

The server will start on `http://localhost:1995` by default.

You should see output similar to:

```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:1995 (Press CTRL+C to quit)
```

### Custom Port

The default port is 1995. To use a different port:

```bash
uv run python src/run.py --port 9000
```

---

## Verification

### Test the API

Open a new terminal and test the API:

```bash
curl http://localhost:1995/health
```

You should receive a response indicating the service is healthy.

### Run Simple Demo

Test the complete workflow with the simple demo:

```bash
# In a new terminal (keep the server running)
uv run python src/bootstrap.py demo/simple_demo.py
```

This will:
1. Store sample conversation messages
2. Wait for indexing
3. Search for relevant memories
4. Display results

If this works, your installation is successful!

---

## Troubleshooting

### Docker Services Not Starting

**Problem**: `docker-compose up -d` fails or services don't start

**Solutions**:
- Check Docker is running: `docker info`
- Check port conflicts: `lsof -i :27017,19200,19530,6379`
- View logs: `docker-compose logs -f`
- Restart services: `docker-compose restart`

### Insufficient Memory

**Problem**: Elasticsearch or Milvus crashes due to OOM

**Solutions**:
- Increase Docker memory limit (Docker Desktop > Preferences > Resources)
- Reduce heap size in docker-compose.yml
- Close other memory-intensive applications

### Python Dependencies Fail

**Problem**: `uv sync` fails with errors

**Solutions**:
- Update uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Clear cache: `uv cache clean`
- Try with verbose output: `uv sync -v`

### API Server Won't Start

**Problem**: Server fails to start or crashes

**Solutions**:
- Check .env file is configured correctly
- Verify all Docker services are running: `docker-compose ps`
- Check logs for specific errors
- Ensure port 1995 is not in use: `lsof -i :1995`

### Connection Errors

**Problem**: Can't connect to MongoDB/Elasticsearch/Milvus

**Solutions**:
- Verify services are running: `docker-compose ps`
- Check connection strings in .env
- Use host ports (27017, 19200, 19530) not container ports
- Test connections individually:
  ```bash
  # MongoDB
  mongosh mongodb://admin:memsys123@localhost:27017

  # Elasticsearch
  curl http://localhost:19200

  # Redis
  redis-cli -h localhost -p 6379 ping
  ```

For more troubleshooting help, see:
- [Docker Setup Guide](DOCKER_SETUP.md)
- [Configuration Guide](../usage/CONFIGURATION_GUIDE.md)
- [GitHub Issues](https://github.com/EverMind-AI/EverMemOS/issues)

---

## Manual Installation (Advanced)

If you prefer not to use Docker, you can install each service manually:

### Required Services

1. **MongoDB 7.0+**
   - See [MongoDB Guide](../usage/MONGODB_GUIDE.md)

2. **Elasticsearch 8.x**
   - Download from [elastic.co](https://www.elastic.co/downloads/elasticsearch)
   - Configure port 9200

3. **Milvus 2.4+**
   - Follow [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md)
   - Configure port 19530

4. **Redis 7.x**
   - Install via package manager or from [redis.io](https://redis.io/download)
   - Configure port 6379

After installing services manually, update connection strings in `.env` accordingly.

---

## Next Steps

Now that EverMemOS is installed, you can:

1. **[Try the Demos](../usage/DEMOS.md)** - Interactive examples showing memory extraction and chat
2. **[Learn the API](../api_docs/memory_api.md)** - Integrate EverMemOS into your application
3. **[Explore Usage Examples](../usage/USAGE_EXAMPLES.md)** - Common usage patterns
4. **[Run Evaluations](../../evaluation/README.md)** - Test on benchmark datasets

---

## See Also

- [Docker Setup Guide](DOCKER_SETUP.md) - Detailed Docker configuration
- [Configuration Guide](../usage/CONFIGURATION_GUIDE.md) - Complete configuration options
- [MongoDB Guide](../usage/MONGODB_GUIDE.md) - MongoDB installation and setup
- [Quick Start (README)](../../README.md#quick-start) - Quick start overview
- [Getting Started for Developers](../dev_docs/getting_started.md) - Development setup
