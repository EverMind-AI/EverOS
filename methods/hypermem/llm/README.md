# Memory Layer LLM Module

This module provides LLM providers for the memory layer functionality, specifically designed to work with OpenRouter API.

## Features

- **OpenAI Provider**: Uses OpenRouter API to access OpenAI models
- **Environment Variable Configuration**: Easy configuration through environment variables
- **Async Support**: Full async/await support for all operations
- **Error Handling**: Comprehensive error handling with custom exceptions
- **Usage Statistics**: Built-in token usage tracking and reporting

## Setup

### 1. Install Dependencies

```bash
pip install langchain-openai langchain-core
```

### 2. Environment Variables

Set the following environment variables:

```bash
# Required
export OPENROUTER_API_KEY="your_openrouter_api_key_here"

# Optional (with defaults)
export OPENROUTER_MODEL="gpt-4o-mini"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
export OPENROUTER_TEMPERATURE="0.3"
export OPENROUTER_MAX_TOKENS="16384"

# Optional: Site attribution
export OPENROUTER_SITE_URL="your_site_url"
export OPENROUTER_APP_NAME="EverMemOS"
```

### 3. Get OpenRouter API Key

1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up for an account
3. Get your API key from the dashboard
4. Add credits to your account for API usage

## Usage

### Basic Usage

```python
from memory_layer.llm import OpenAIProvider

# Create provider with environment variables
provider = OpenAIProvider.from_env()

# Generate text
response = await provider.generate("Hello, how are you?")
print(response)
```

### Custom Configuration

```python
from memory_layer.llm import OpenAIProvider

# Create provider with custom settings
provider = OpenAIProvider(
    model="gpt-4o",
    api_key="your_api_key",
    temperature=0.7,
    max_tokens=2048
)

# Generate text
response = await provider.generate("Explain quantum computing", temperature=0.5)
print(response)
```

### Factory Functions

```python
from memory_layer.llm import create_provider, create_provider_from_env

# Create from environment variables
provider = create_provider_from_env("openai")

# Create with custom settings
provider = create_provider("openai", model="gpt-4o", temperature=0.7)
```

### Test Connection

```python
# Test if the provider can connect to the API
is_connected = await provider.test_connection()
if is_connected:
    print("Connection successful!")
else:
    print("Connection failed!")
```

## API Reference

### OpenAIProvider

#### Constructor Parameters

- `model` (str): Model name (default: "gpt-4o-mini")
- `api_key` (str): OpenRouter API key (default: from OPENROUTER_API_KEY env var)
- `base_url` (str): API base URL (default: "https://openrouter.ai/api/v1")
- `temperature` (float): Sampling temperature (default: 0.3)
- `max_tokens` (int): Maximum tokens to generate (default: 16384)

#### Methods

- `async generate(prompt: str, temperature: float | None = None) -> str`: Generate text response
- `async test_connection() -> bool`: Test API connection
- `from_env(**kwargs) -> OpenAIProvider`: Create provider from environment variables

## Error Handling

The module provides custom exception handling:

```python
from memory_layer.llm import LLMError

try:
    response = await provider.generate("Hello")
except LLMError as e:
    print(f"LLM Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Usage Statistics

The provider automatically tracks and displays usage statistics:

```
🤖 OpenRouter API调用统计:
   模型: gpt-4o-mini
   耗时: 1.23秒
   输入tokens: 15
   输出tokens: 42
   总tokens: 57
```

## Supported Models

The provider supports all OpenAI models available through OpenRouter, including:

- gpt-4o-mini (recommended for most use cases)
- gpt-4o
- gpt-4-turbo
- gpt-3.5-turbo
- And many more...

Check [OpenRouter's model list](https://openrouter.ai/models) for the complete list of available models.
