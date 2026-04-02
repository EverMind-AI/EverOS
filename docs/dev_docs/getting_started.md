# Intelligent Memory System - Quick Start Guide

This guide will help you quickly set up and launch the Intelligent Memory System project.

## 📋 Table of Contents

- [Requirements](#requirements)
- [Install Dependencies](#install-dependencies)
- [Environment Configuration](#environment-configuration)
- [Start Services](#start-services)
- [Run Test Scripts](#run-test-scripts)
- [Common Issues](#common-issues)

## 🔧 Requirements

### System Requirements
- **Operating System**: macOS, Linux, Windows
- **Python Version**: 3.10+
- **Package Manager**: uv (recommended)

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
# Clone the project locally
git clone <project-url>
cd memsys_opensource

# View project structure
ls -la
```

### 3. Install Project Dependencies

```bash
# Sync dependencies using uv (recommended)
# uv will automatically create a virtual environment and install all dependencies
uv sync

# Verify installation
uv run python --version
```

## ⚙️ Environment Configuration

### 1. Create Environment Configuration File

```bash
# Copy environment variable template
cp env.template .env

# Open .env file with an editor
vim .env  # or use your preferred editor
```

### 2. Configure Required Environment Variables

Edit the `.env` file and fill in the actual configuration values:

#### LLM Configuration
```bash
# LLM configuration
LLM_PROVIDER=openai
LLM_MODEL=google/gemini-2.5-flash
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-your-api-key
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=16384
```

#### DeepInfra Configuration (for Embedding and Rerank)
```bash
# DeepInfra Embedding
VECTORIZE_API_KEY=your-deepinfra-key
VECTORIZE_BASE_URL=https://api.deepinfra.com/v1/openai
VECTORIZE_MODEL=Qwen/Qwen3-Embedding-4B
VECTORIZE_TIMEOUT=30
VECTORIZE_MAX_RETRIES=3
VECTORIZE_BATCH_SIZE=10
VECTORIZE_MAX_CONCURRENT=5
VECTORIZE_ENCODING_FORMAT=float
VECTORIZE_DIMENSIONS=1024

# DeepInfra Rerank
RERANK_BASE_URL=https://api.deepinfra.com/v1/inference
RERANK_MODEL=Qwen/Qwen3-Reranker-4B
RERANK_TIMEOUT=30
RERANK_MAX_RETRIES=3
RERANK_BATCH_SIZE=10
RERANK_MAX_CONCURRENT=5
```

#### Database Configuration
```bash
# Redis Configuration
REDIS_HOST=127.0.0.1
REDIS_PORT=6379
REDIS_DB=8
REDIS_SSL=false

# MongoDB Configuration
MONGODB_HOST=your-mongodb-host
MONGODB_PORT=27017
MONGODB_USERNAME=your_username
MONGODB_PASSWORD=your_password
MONGODB_DATABASE=your_database_name
MONGODB_URI_PARAMS="socketTimeoutMS=15000&authSource=admin"

# Elasticsearch Configuration
ES_HOSTS=https://your-elasticsearch-host:9200
ES_USERNAME=elastic
ES_PASSWORD=your_password
ES_VERIFY_CERTS=true
SELF_ES_INDEX_NS=your-namespace

# Milvus Vector Database Configuration
MILVUS_HOST=your-milvus-host
MILVUS_PORT=19530
SELF_MILVUS_COLLECTION_NS=your_namespace
```

#### Environment and Logging Configuration
```bash
LOG_LEVEL=DEBUG
ENV=dev
PYTHONASYNCIODEBUG=1
```

### 3. Obtain API Keys

#### OpenRouter API Key
1. Visit [OpenRouter](https://openrouter.ai/)
2. Register an account and create an API key
3. Fill the key into `LLM_API_KEY` in the `.env` file

#### DeepInfra API Key
1. Visit [DeepInfra](https://deepinfra.com/)
2. Register an account and create an API key
3. Fill the key into `VECTORIZE_API_KEY` in the `.env` file

## 🚀 Start Services

### 1. Start Web Service (REST API)

Start the main application service, providing REST API endpoints:

```bash
# Basic startup (using default port 1995)
uv run python src/run.py --port 1995

# Start with specified log level
LOG_LEVEL=DEBUG uv run python src/run.py

# Start with specified port
uv run python src/run.py --port 8080

# Specify host and port
uv run python src/run.py --host 0.0.0.0 --port 8080

# Use custom environment file
uv run python src/run.py --env-file .env.production
```

#### Startup Parameter Description
- `--host`: Server listening address (default: 0.0.0.0)
- `--port`: Server port (default: 1995)
- `--env-file`: Environment variable file path (default: .env)
- `--mock`: Enable Mock mode (for testing and development)

#### Successful Startup Output Example
```
🚀 Starting Memory System v1.0.0
📝 Memory System Main Application
🌟 Startup Parameters:
  📡 Host: 0.0.0.0
  🔌 Port: 1995
  📄 Env File: .env
  🎭 Mock Mode: False
  🔧 LongJob Mode: Disabled
🚀 Initializing dependency injection container...
✅ Dependency injection setup complete
🔄 Registering async tasks...
✅ Async task registration complete
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:1995 (Press CTRL+C to quit)
```

### 2. Verify Service Startup

After the service starts, you can verify it by:

```bash
# Access API documentation
curl http://localhost:1995/docs

# Or open in browser
open http://localhost:1995/docs
```

### 3. Start Long Job Consumer (Optional)

If you need to start an async task processor (such as Kafka consumer):

```bash
# Start Kafka consumer
uv run python src/run.py --longjob kafka_consumer

# Stop task (in another terminal)
pkill -9 -f 'src/run.py'
```

## 🧪 Run Test Scripts

### Bootstrap Script Introduction

`bootstrap.py` is a universal script runner that automatically handles:
- Python path setup
- Environment variable loading
- Dependency injection container initialization
- Application context management

Using Bootstrap allows you to run any test script **without cognitive burden**.

### Basic Usage

```bash
# Basic syntax
uv run python src/bootstrap.py [script-path] [script-arguments...]

# Run test script
uv run python src/bootstrap.py tests/test_convert_rest.py

# Run script with arguments
uv run python src/bootstrap.py tests/my_test.py --verbose

# Run in Mock mode
uv run python src/bootstrap.py tests/my_test.py --mock

# Use custom environment file
uv run python src/bootstrap.py tests/my_test.py --env-file .env.test
```

### Practical Examples

#### 1. Run Evaluation Scripts
```bash
# Run LoCoMo evaluation stage 1
uv run python src/bootstrap.py evaluation/locomo_evaluation/stage1_memcells_extraction.py

# Run other evaluation stages
uv run python src/bootstrap.py evaluation/locomo_evaluation/stage2_index_building.py
uv run python src/bootstrap.py evaluation/locomo_evaluation/stage3_memory_retrieval.py
```

#### 2. Run Demo Scripts
```bash
# Run memory extraction demo
uv run python src/bootstrap.py demo/extract_memory.py

# Run memory chat demo
uv run python src/bootstrap.py demo/chat_with_memory.py
```

#### 3. Run Unit Tests
```bash
# Run test template (learn how to use DI and MongoDB)
uv run python src/bootstrap.py tests/bootstrap_test_template.py

# Run custom tests
uv run python src/bootstrap.py unit_test/my_unit_test.py
```

### Bootstrap Command-Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--env-file` | Specify environment variable file | `--env-file .env.test` |
| `--mock` | Enable Mock mode | `--mock` |

### Bootstrap Working Principle

1. **Automatic Environment Setup**: Load `.env` file, set Python path
2. **Initialize Dependency Injection**: Start DI container, register all components
3. **Start Application Context**: Initialize database connections, caches, etc.
4. **Execute Target Script**: Run your script in complete application context
5. **Clean Up Resources**: Automatically clean up after script execution

## 🐛 Development Debugging

### 1. Mock Mode

During development and testing, you can enable Mock mode to simulate external dependencies:

```bash
# Method 1: Use command-line argument
uv run python src/run.py --mock

# Method 2: Set environment variable
export MOCK_MODE=true
uv run python src/run.py

# Method 3: Configure in .env file
# MOCK_MODE=true
```

### 2. Debug Logging

```bash
# Set verbose log level
export LOG_LEVEL=DEBUG
uv run python src/run.py

# Or specify directly in command line
LOG_LEVEL=DEBUG uv run python src/run.py
```

### 3. Development Environment Configuration

Create development-specific environment configuration:

```bash
# Create development environment configuration
cp .env .env.dev

# Edit development configuration
vim .env.dev
```

Set development-related configurations in `.env.dev`:
```bash
# Development mode
ENV=dev
DEBUG=true
LOG_LEVEL=DEBUG
MOCK_MODE=true

# Local services
MONGODB_HOST=localhost
REDIS_HOST=localhost
ES_HOSTS=http://localhost:19200
MILVUS_HOST=localhost
```

Start with development configuration:
```bash
uv run python src/run.py --env-file .env.dev
```

## ❓ Common Issues

### 1. uv Related Issues

#### Issue: uv sync fails
```bash
# Solution: Clean cache and retry
uv cache clean
uv sync

# Or use pip as fallback
pip install -e .
```

#### Issue: uv command not found
```bash
# Ensure uv is installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Reload shell configuration
source ~/.bashrc  # or source ~/.zshrc
```

### 2. Environment Configuration Issues

#### Issue: .env file not found
```bash
# Ensure .env file exists
ls -la .env

# If it doesn't exist, copy template
cp env.template .env
```

#### Issue: Environment variables not taking effect
```bash
# Check .env file format
cat .env | grep -v "^#" | grep -v "^$"

# Ensure no extra spaces and quotes
```

### 3. Database Connection Issues

#### Issue: MongoDB connection fails
```bash
# Check if MongoDB is running
# macOS
brew services list | grep mongodb

# Linux
systemctl status mongod

# Check connection configuration
echo $MONGODB_HOST
echo $MONGODB_PORT
```

#### Issue: Redis connection fails
```bash
# Check if Redis is running
redis-cli ping

# If not running, start Redis
# macOS
brew services start redis

# Linux
sudo systemctl start redis
```

### 4. Startup Failure Issues

#### Issue: Port already in use
```bash
# Check port usage
lsof -i :1995

# Start with different port
uv run python src/run.py --port 8080
```

#### Issue: Module import error
```bash
# Ensure executing in project root directory
pwd

# Reinstall dependencies
uv sync --reinstall
```

### 5. Bootstrap Run Issues

#### Issue: Script path not found
```bash
# Ensure using correct relative path
ls -la evaluation/locomo_evaluation/stage1_memcells_extraction.py

# Or use absolute path
uv run python src/bootstrap.py /path/to/your/script.py
```

#### Issue: Script execution error
```bash
# View detailed error information
LOG_LEVEL=DEBUG uv run python src/bootstrap.py your_script.py

# Test with Mock mode
uv run python src/bootstrap.py your_script.py --mock
```

## 🎯 Next Steps

Now you have successfully set up and launched the Intelligent Memory System! Next, you can:

1. **Read Development Guide**: Check [development_guide.md](development_guide.md) to learn about project architecture and best practices
2. **Explore Bootstrap**: Check [bootstrap_usage.md](bootstrap_usage.md) for in-depth understanding of script runner
3. **View API Documentation**: Visit http://localhost:1995/docs to learn about available API endpoints
4. **Run Example Code**: Try running example scripts in the `demo/` directory

## 📞 Get Help

If you encounter issues, you can get help through:

1. **View Logs**: Use `LOG_LEVEL=DEBUG` to view detailed logs
2. **Check Configuration**: Confirm `.env` file is configured correctly
3. **Read Documentation**: Read relevant technical documentation
4. **Submit Issue**: Report issues in the project repository

---

**Enjoy using the system!** 🎉

