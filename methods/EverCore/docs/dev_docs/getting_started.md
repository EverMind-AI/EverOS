# Intelligent Memory System - Quick Start Guide

This guide will help you quickly set up and launch the Intelligent Memory System project.

## 📋 Table of Contents

- [Requirements](#requirements)
- [Install Dependencies](#install-dependencies)
- [Environment Configuration](#environment-configuration)
- [Start Infrastructure](#start-infrastructure)
- [Start Services](#start-services)
- [VSCode Debug Launch](#vscode-debug-launch)
- [Run Test Scripts](#run-test-scripts)
- [Development Debugging](#development-debugging)
- [Common Issues](#common-issues)

## 🔧 Requirements

### System Requirements
- **Operating System**: macOS, Linux, Windows
- **Python Version**: 3.12+
- **Package Manager**: uv (recommended)
- **Docker**: Required for local infrastructure (MongoDB, Elasticsearch, Milvus, Redis)

### Required External Services
- **MongoDB**: For storing memory data
- **Redis**: For caching and task queues
- **Elasticsearch**: For full-text search
- **Milvus**: For vector retrieval

## 📦 Install Dependencies

### 1. Install uv

uv is a fast Python package manager, highly recommended.

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Verify installation
uv --version
```

### 2. Clone the Project

```bash
git clone <project-url>
cd memsys_opensource
```

### 3. Install Project Dependencies

```bash
# uv will automatically create a virtual environment and install all dependencies
uv sync
```

## ⚙️ Environment Configuration

### 1. Create Environment Configuration File

```bash
cp env.template .env
```

### 2. Configure Required Environment Variables

Edit the `.env` file and fill in actual values for the following required items:

#### LLM Configuration

```bash
# LLM provider and model
LLM_PROVIDER=openrouter
LLM_MODEL=x-ai/grok-4-fast
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=32768

# OpenRouter API key (or your preferred provider)
OPENROUTER_API_KEY=sk-or-v1-your-key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Supported providers: `openrouter`, `openai`. Set the corresponding `{PROVIDER}_API_KEY` and `{PROVIDER}_BASE_URL`.

#### Vectorize (Embedding) Configuration

```bash
# Primary provider: vllm (self-deployed) or deepinfra (commercial API)
VECTORIZE_PROVIDER=deepinfra
VECTORIZE_API_KEY=your-deepinfra-key
VECTORIZE_BASE_URL=https://api.deepinfra.com/v1/openai
VECTORIZE_MODEL=Qwen/Qwen3-Embedding-4B
VECTORIZE_DIMENSIONS=1024
```

#### Rerank Configuration

```bash
# Primary provider: vllm or deepinfra
RERANK_PROVIDER=deepinfra
RERANK_API_KEY=your-deepinfra-key
RERANK_BASE_URL=https://api.deepinfra.com/v1/inference
RERANK_MODEL=Qwen/Qwen3-Reranker-4B
```

#### Database Configuration

The defaults in `env.template` match the Docker Compose services. If using local Docker, no changes needed:

```bash
# Tenant ID (required, use t_{yourname} for local dev)
TENANT_SINGLE_TENANT_ID=t_yourname

# MongoDB (default: local Docker)
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USERNAME=admin
MONGODB_PASSWORD=memsys123

# Elasticsearch (default: local Docker, port 19200)
ES_HOSTS=http://localhost:19200
ES_VERIFY_CERTS=false

# Milvus (default: local Docker)
MILVUS_HOST=localhost
MILVUS_PORT=19530

# Redis (default: local Docker)
REDIS_HOST=localhost
REDIS_PORT=6379
```

> **Note**: `TENANT_SINGLE_TENANT_ID` is required. All storage resources will be prefixed with this value (e.g., `t_yourname_memsys`). Use `t_{yourname}` to avoid conflicts with other developers on shared infrastructure.

### 3. Obtain API Keys

- **OpenRouter**: Register at [openrouter.ai](https://openrouter.ai/) and create an API key
- **DeepInfra**: Register at [deepinfra.com](https://deepinfra.com/) and create an API key (for Embedding + Rerank)

## 🐳 Start Infrastructure

Start MongoDB, Elasticsearch, Milvus, and Redis via Docker Compose:

```bash
docker-compose up -d
```

Wait for all services to be healthy (about 30-60 seconds on first run). Check status:

```bash
docker-compose ps
```

## 🚀 Start Services

### Start Web Service (REST API)

```bash
# Default port 1995
uv run python src/run.py

# Specify port
uv run python src/run.py --port 8080

# Enable debug logging
LOG_LEVEL=DEBUG uv run python src/run.py
```

#### Startup Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--host` | Listening address | `0.0.0.0` |
| `--port` | Port | `1995` |
| `--env-file` | Environment variable file | `.env` |
| `--mock` | Enable Mock mode | disabled |

After startup, visit API documentation: `http://localhost:1995/docs`

### Start Task Worker (Optional)

For async task processing:

```bash
uv run .venv/bin/arq task.WorkerSettings
```

### Start Long Job (Optional)

For persistent background processes (e.g., Kafka consumer):

```bash
uv run python src/run.py --longjob kafka_consumer
```

## 🖥️ VSCode Debug Launch

The project includes pre-configured launch configurations in `.vscode/launch.json`. Open the project in VSCode and press `F5` or use the **Run and Debug** panel:

| Configuration | Description |
|---------------|-------------|
| `Python 调试程序: run` | Start API service (most common) |
| `Python 调试程序: task` | Start Task Worker |
| `Python 调试程序: longjob` | Start Long Job (e.g., Kafka consumer) |
| `Python 调试程序: run_this_file` | Run the currently open file via bootstrap |

All configurations automatically read the `.env` file and support full breakpoint debugging.

## 🧪 Run Test Scripts

### Bootstrap Script

`src/bootstrap.py` is a universal script runner that handles environment setup, DI initialization, and application context automatically.

```bash
# Basic usage
uv run python src/bootstrap.py <script-path> [args...]

# Examples
uv run python src/bootstrap.py demo/extract_memory.py
uv run python src/bootstrap.py demo/chat_with_memory.py

# Enable mock mode
uv run python src/bootstrap.py your_script.py --mock

# Use custom env file
uv run python src/bootstrap.py your_script.py --env-file .env.test
```

### Run Unit Tests

```bash
PYTHONPATH=src pytest tests/
PYTHONPATH=src pytest tests/path/to/test_file.py
```

## 🐛 Development Debugging

### Mock Mode

Use Mock mode to bypass external dependencies during development:

```bash
# Command line
uv run python src/run.py --mock

# Environment variable
export MOCK_MODE=true
uv run python src/run.py
```

### Debug Logging

```bash
LOG_LEVEL=DEBUG uv run python src/run.py
```

## ❓ Common Issues

### uv sync fails

```bash
uv cache clean
uv sync
```

### .env file not found

```bash
cp env.template .env
```

### Docker services not starting

```bash
# Check logs
docker-compose logs mongodb
docker-compose logs elasticsearch

# Restart specific service
docker-compose restart milvus
```

### Port already in use

```bash
# Check port usage
lsof -i :1995

# Use a different port
uv run python src/run.py --port 8080
```

### Module import error

```bash
# Ensure running from project root
pwd  # Should be .../memsys_opensource

# Reinstall dependencies
uv sync --reinstall
```

### Enterprise package missing after uv sync

After running `uv sync` in `memsys_opensource`, enterprise's editable install is lost. Re-install:

```bash
source .venv/bin/activate
cd ../memsys_enterprise
pip install -e .
```

## 🎯 Next Steps

1. **Read Development Guide**: Check [development_guide.md](development_guide.md) for architecture and best practices
2. **View API Documentation**: Visit `http://localhost:1995/docs` for available API endpoints
3. **Run Demos**: Try example scripts in the `demo/` directory
4. **Read Bootstrap Guide**: See [bootstrap_usage.md](bootstrap_usage.md) for script runner details
