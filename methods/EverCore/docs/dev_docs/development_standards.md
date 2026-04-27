# Development Standards

This document introduces various standards and best practices in the project development process to help the team maintain code quality and collaboration efficiency.

---

## 🚀 TL;DR (Core Principles)

### Quick Start for Newcomers (2 Steps)
```bash
make dev-setup              # One-click dev environment setup (sync deps + install hooks)
```

### Core Conventions

**📦 Dependency Management**  
Use `uv add/remove` to manage dependencies, avoid direct `pip install` to maintain consistency of dependency lock files

**🎨 Code Style**  
Pre-commit checks run automatically on commit (black/ruff/isort) to keep code style consistent

**⚡️ Full Async Architecture**  
Single Event Loop, use `async/await` for I/O operations, discuss with development lead before using threads/processes

**🚫 No I/O in Loops**  
Prohibit database access and API calls in for loops, use batch operations instead

**🕐 Timezone Awareness**  
All time fields must carry timezone information. Input without timezone is treated as the timezone configured by environment variable `TZ` (default UTC). Do not use `datetime.datetime.now()`, must use utility functions from `common_utils/datetime_utils.py`

**📥 Import Standards**  
- PYTHONPATH management: Project module import starting paths (src/tests/demo etc.) need unified management, communicate with development lead before changes
- Prefer absolute imports (e.g. `from core.memory import MemoryManager`), avoid relative imports (e.g. `from ...core import`)

**📝 __init__.py Standards**  
Not recommended to write any code in `__init__.py`, keep it empty

**🌿 Branch Standards**  
`dev` for daily development, `release/YYMMDD` for versioned releases, `long/xxx` for long-term feature development, `hotfix` for emergency fixes

**🔀 Unified Branch Merge Handling**  
Merging `long/xxx` to `dev`, cutting `release` from `dev`, merging `release` back to `dev` needs to be handled uniformly by development or operations lead

**📤 MR Standards**
- Keep code commits small, iterate quickly, avoid submitting too much code at once
- Each commit should be runnable, do not submit work-in-progress or broken code
- Data migration scripts, dependency changes, infrastructure code changes, merging release branches must go through Code Review

**💾 Data Migration Standards**  
For new features involving data fixes or Schema migration, discuss feasibility and implementation timing with development and operations as early as possible

**🏛️ Data Access Standards**  
All database, search engine and other external storage read/write operations must be converged to infra layer repository methods. Direct calls in business layer are prohibited

**🎯 Minimal Changes**  
Minimize code changes when implementing requirements, avoid large-scale refactoring, prioritize incremental development. Do not over-engineer, keep it simple, efficient, and maintainable

**💬 Comment Standards**  
Always add sufficient comments (function-level + step-level), ensure reviewers can quickly understand code intent

**📖 API Documentation Sync**  
When modifying API interfaces, must synchronize updates to API documentation comments, schema definition files, and auto-generated documentation files

**📄 Documentation Standards**  
Use markdown format, place in docs directory. Small issues don't need documentation, just add comments in code

**🌍 Internationalization Standards**  
For international team communication and collaboration, code comments and documentation should be written in English

### 📖 Quick Navigation

- Don't know how to install dependencies? → [Dependency Management Standards](#-dependency-management-standards)
- Need database/middleware configuration? → [Development Environment Configuration Standards](#-development-environment-configuration-standards)
- Always getting errors before commit? → [Code Style Standards](#-code-style-standards)
- How to write code comments? → [Comment Standards](#-comment-standards)
- What to do after changing API? → [API Specification Sync](#-api-specification-sync)
- Not sure if you can use threads? → [Async Programming Standards](#️-async-programming-standards)
- Can I do database queries in loops? → [Prohibit I/O Operations in for Loops](#7-prohibit-io-operations-in-for-loops-)
- How to handle time fields? → [Timezone Awareness Standards](#-timezone-awareness-standards)
- Where should database queries be written? → [Data Access Standards](#️-data-access-standards)
- Import path errors? → [Import Standards](#-import-standards)
- How to name module introduction files? → [Module Introduction File Naming](#-module-introduction-file-naming)
- Don't know which branch to use? → [Branch Management Standards](#-branch-management-standards)
- How to submit code/Need to submit MR? → [MR Standards](#-mr-standards)
- Need data migration? → [Data Migration and Schema Change Process](#data-migration-and-schema-change-process)

---

## 📋 Table of Contents

- [TL;DR (Core Principles)](#-tldr-core-principles)
- [Dependency Management Standards](#-dependency-management-standards)
- [Development Environment Configuration Standards](#-development-environment-configuration-standards)
- [Code Style Standards](#-code-style-standards)
- [Comment Standards](#-comment-standards)
- [API Specification Sync](#-api-specification-sync)
- [Documentation Standards](#-documentation-standards)
- [Async Programming Standards](#️-async-programming-standards)
- [Timezone Awareness Standards](#-timezone-awareness-standards)
- [Data Access Standards](#️-data-access-standards)
- [Import Standards](#-import-standards)
  - [PYTHONPATH Management](#pythonpath-management)
  - [Prefer Absolute Imports](#prefer-absolute-imports)
  - [__init__.py Usage Standards](#__init__py-usage-standards)
- [Module Introduction File Naming](#-module-introduction-file-naming)
- [Internationalization Standards](#-internationalization-standards)
- [Branch Management Standards](#-branch-management-standards)
- [MR Standards](#-mr-standards)
- [Code Review Process](#-code-review-process)
  - [Data Migration and Schema Change Process](#data-migration-and-schema-change-process)

---

## 📦 Dependency Management Standards

### Using uv for Dependency Management

**💡 Important Note: Recommended to use uv for dependency management**

The project uses `uv` as the dependency management tool. It's recommended to avoid using `pip install` directly for the following reasons:

- Dependency versions may be inconsistent
- `uv.lock` file cannot be automatically updated
- Team member environments may differ
- May affect production environment deployment

### Correct Operations

#### 1. Install/Sync Dependencies

```bash
# Sync all dependencies (first install or after updates)
uv sync --group dev-full
```

#### 2. Add New Dependencies

```bash
# Add production dependency
uv add <package-name>

# Add development dependency
uv add --dev <package-name>

# Specify version
uv add <package-name>==<version>
```

#### 3. Remove Dependencies

```bash
uv remove <package-name>
```

#### 4. Update Dependencies

```bash
# Update all dependencies
uv sync --upgrade

# Update specific dependency
uv add <package-name> --upgrade
```

### Related Documentation

For detailed dependency management guide, refer to: [project_deps_manage.md](./project_deps_manage.md)

---

## 🔧 Development Environment Configuration Standards

### Environment Configuration Description

The project depends on various databases and middleware. To ensure consistency and security of development environments, these configurations are uniformly managed and distributed by the operations team.

#### Configuration Items

Development environment typically needs the following configurations:

**Database Configuration**
- MongoDB connection information
- PostgreSQL connection information
- Redis connection information

**Middleware Configuration**
- Kafka connection configuration
- ElasticSearch connection configuration
- Other message queues or cache services

**Third-party Service Configuration**
- API keys and access credentials
- Object storage configuration
- Other external service credentials

### How to Get Configuration

#### 1. New Employee Onboarding

New developers joining the project, please follow this process to get configurations:

1. **Contact operations lead** (see contact information at the end of document)
2. **State your needs**:
   - Your name and role
   - Environment needed (development/testing)
   - Specific services to access
3. **Receive configuration**: Operations lead will provide configuration files or environment variables
4. **Local configuration**: Place configuration information in project's `config.json` or `.env` file (Note: these files are in `.gitignore`, won't be committed to repository)

#### 2. Configuration File Location

```bash
# Configuration files in project root (do not commit to git)
config.json          # Main configuration file
.env                 # Environment variable configuration
env.template         # Configuration template (reference, need to fill in real values)
```

#### 3. Environment Variable Examples

Reference `env.template` file, your `.env` file typically contains the following types of configuration:

```bash
# Tenant (required for local dev)
TENANT_SINGLE_TENANT_ID=t_yourname

# MongoDB
MONGODB_URI=mongodb://...

# Redis
REDIS_HOST=...
REDIS_PORT=...
REDIS_PASSWORD=...

# Kafka
KAFKA_BOOTSTRAP_SERVERS=...

# ElasticSearch
ES_HOST=...
ES_PORT=...
```

### Configuration Management Notes

#### ⚠️ Security Standards

1. **Prohibit committing sensitive configuration**
   - All configuration files containing passwords, keys, tokens must not be committed to git
   - Check `.gitignore` includes configuration files before committing
   - Using pre-commit hook can help detect sensitive information

2. **Configuration file permissions**
   - Local configuration files should have appropriate permissions (only current user readable)
   - Do not paste configuration content in public places (like chat records, documents)

3. **Configuration update notifications**
   - If configuration is updated, operations team will notify relevant developers
   - Update local configuration promptly after receiving notification

#### 🔄 Configuration Change Process

If you need to:
- Add new configuration items
- Modify configuration structure
- Add new environments or services

**Recommended process**:

1. **Discuss with development lead**: Confirm necessity and impact scope of configuration changes
2. **Contact operations lead**: Explain configuration needs and reasons for changes
3. **Update configuration template**: Update `env.template` and related documentation
4. **Team notification**: Notify all developers to sync update local configuration

### Different Environment Description

| Environment | Purpose | Configuration Source | Notes |
|-------------|---------|---------------------|-------|
| **Development** | Local development and debugging | Provided by operations | Usually connects to development database, data can be freely tested |
| **Testing** | Integration and functional testing | Auto-deployed configuration | Connects to test database, data periodically reset |
| **Production** | Live running services | Strictly controlled by operations | Only operations and authorized personnel can access |

**Note**: Developers usually only need development environment configuration. Testing and production environment configurations are managed by CI/CD and operations team.

---

## 🎨 Code Style Standards

### Pre-commit Hook Configuration

The project uses `pre-commit` to unify code style. It's recommended to install pre-commit hook after first cloning the project.

#### Installation Steps

```bash
# One-click dev environment setup (sync deps + install hooks)
make dev-setup
```

> **Tip**: `make dev-setup` automatically runs `uv sync --dev` and installs pre-commit hooks. If you only need to install hooks separately, run `make setup-hooks`.

#### Functions

Pre-commit hook will automatically execute the following checks before each commit:

- **Code formatting**: Format Python code using black/ruff
- **Import sorting**: Sort import statements using isort
- **Code checking**: Code quality check using ruff/flake8
- **Type checking**: Type check using pyright/mypy
- **YAML/JSON format**: Check configuration file format
- **Trailing whitespace**: Remove extra whitespace at end of files

#### Manual Check

```bash
# Run check on all files
pre-commit run --all-files

# Run check on staged files
pre-commit run
```

---

## 💬 Comment Standards

### Core Principle

**💡 Important Note: Always add sufficient comments**

Good comments help team members quickly understand code intent, improving maintainability and Code Review efficiency.

### Comment Requirements

#### 1. Function-level Comments (Google-style Docstring)

Every function/method should have a clear **Google-style docstring** explaining:

- **Description**: What the function does
- **Args**: Type and purpose of each parameter
- **Returns**: Return value type and meaning
- **Raises**: Exceptions that may be thrown (if applicable)

```python
# ✅ Recommended: Complete function-level comments
async def fetch_user_memories(
    user_id: str,
    limit: int = 100,
    include_archived: bool = False
) -> list[Memory]:
    """
    Fetch user's memory list.
    
    Args:
        user_id: User unique identifier
        limit: Maximum number of memories to return, default 100
        include_archived: Whether to include archived memories, default False
    
    Returns:
        User's memory list, sorted by creation time in descending order
    
    Raises:
        UserNotFoundError: When user does not exist
    """
    ...
```

#### 2. Step-level Comments

In complex business logic, add comments at key steps to explain the purpose of each step:

```python
# ✅ Recommended: Add comments at key steps
async def process_memory_extraction(raw_data: dict) -> Memory:
    # 1. Validate input data integrity
    validated_data = validate_input(raw_data)
    
    # 2. Extract key information (people, events, time, etc.)
    extracted_info = await extract_key_information(validated_data)
    
    # 3. Generate vector embedding for subsequent retrieval
    embedding = await generate_embedding(extracted_info.content)
    
    # 4. Build memory object and persist
    memory = Memory(
        content=extracted_info.content,
        embedding=embedding,
        metadata=extracted_info.metadata
    )
    
    return memory
```

#### 3. Complex Logic Explanation

For complex algorithms, business rules, or non-intuitive code, add detailed explanations:

```python
# ✅ Recommended: Explain complex business rules
def calculate_memory_score(memory: Memory, query: str) -> float:
    """Calculate relevance score between memory and query"""
    # Base similarity score (cosine similarity)
    base_score = cosine_similarity(memory.embedding, query_embedding)
    
    # Time decay factor: newer memories have higher weight
    # Using exponential decay with half-life of 30 days
    days_old = (now - memory.created_at).days
    time_decay = math.exp(-0.693 * days_old / 30)
    
    # Importance weighting: memories marked as important get 50% boost
    importance_boost = 1.5 if memory.is_important else 1.0
    
    return base_score * time_decay * importance_boost
```

### Comment Style

- Use Chinese or English consistently within the same project/module
- Comments should be concise and clear, avoid redundancy
- Keep comments updated when code changes
- Don't comment obvious code

```python
# ❌ Not recommended: Redundant comment
i = i + 1  # increment i by 1

# ✅ Recommended: Explain "why" not "what"
i = i + 1  # Skip header row, start processing from data rows
```

### Checklist

Before submitting code, confirm:

- [ ] All public functions/methods have docstrings
- [ ] Complex business logic has step-level comments
- [ ] Non-intuitive code has explanatory comments
- [ ] Comments are in sync with code, no outdated comments
- [ ] Reviewers can quickly understand code intent

---

## 📖 API Specification Sync

### Core Principle

**💡 Important Note: Must synchronize API documentation when modifying API interfaces**

API documentation is the key basis for frontend-backend collaboration and service integration. Inconsistency between documentation and actual API leads to integration issues and debugging difficulties.

### Sync Requirements

When modifying API interfaces, must complete the following sync operations:

#### 1. Update API Documentation Comments

Ensure code API documentation comments match actual behavior:

```python
# ✅ Recommended: Keep documentation comments consistent with actual API
from fastapi import APIRouter, Query

router = APIRouter()

@router.get("/memories/{memory_id}")
async def get_memory(
    memory_id: str,
    include_embedding: bool = Query(False, description="Whether to return vector embedding")
) -> MemoryResponse:
    """
    Get detailed information of specified memory.
    
    - **memory_id**: Memory unique identifier
    - **include_embedding**: Whether to include vector embedding data in response
    
    Returns:
        MemoryResponse: Memory details including content, metadata, etc.
    
    Raises:
        404: Memory not found
        403: No permission to access this memory
    """
    ...
```

#### 2. Update Schema Definition Files

If API request/response structure changes, update related schema definitions:

```python
# Update Pydantic model
class MemoryResponse(BaseModel):
    """Memory response model"""
    id: str = Field(..., description="Memory unique identifier")
    content: str = Field(..., description="Memory content")
    created_at: datetime = Field(..., description="Creation time")
    # When adding new fields, add clear descriptions
    embedding: list[float] | None = Field(None, description="Vector embedding, only returned on request")
```

#### 3. Regenerate API Documentation Files

If the project uses auto-generated API documentation (e.g., OpenAPI/Swagger), ensure regeneration:

```bash
# Example: Regenerate OpenAPI documentation
python scripts/generate_openapi.py

# Or ensure FastAPI auto-generated docs are up to date
# Visit /docs or /redoc to verify
```

#### 4. Notify Stakeholders

If it's a major API change, notify frontend and other dependent service developers.

### Checklist

Before submitting API changes, confirm:

- [ ] API documentation comments updated and consistent with actual behavior
- [ ] Schema definition files (Pydantic models, etc.) updated
- [ ] Auto-generated API documentation files regenerated
- [ ] Frontend and other services can develop based on latest API specification
- [ ] If breaking changes, stakeholders have been notified

---

## 📄 Documentation Standards

### Core Principle

**💡 Important Note: Do not over-generate documentation**

Documentation is an important supplement to code, but excessive documentation increases maintenance burden. Follow the "necessary and sufficient" principle.

### When Documentation is Needed

| Scenario | Need Documentation | Notes |
|----------|-------------------|-------|
| Small bug fix | ❌ No | Just explain in code comments |
| Small feature optimization | ❌ No | Explain in commit message and code comments |
| New API endpoint | ⚠️ Depends | API doc comments required, separate doc depends on complexity |
| New module/component | ✅ Yes | Write module introduction documentation |
| Large-scale refactoring | ✅ Yes | Document reasons, approach and impact |
| Architecture design changes | ✅ Yes | Document design decisions and architecture description |
| Complex business processes | ✅ Yes | Write process documentation |

### Documentation Format Requirements

- **Format**: Use Markdown (`.md`) format
- **Syntax**: Follow standard Markdown syntax

### Documentation Location

```
project_root/
├── docs/                        # Documentation root
│   ├── api_docs/               # API documentation
│   │   └── memory_api.md
│   ├── dev_docs/               # Development documentation
│   │   └── development_standards.md
│   ├── architecture/           # Architecture documentation
│   │   └── system_design.md
│   └── guides/                 # User guides
│       └── getting_started.md
```

### Naming Convention

- **Format**: `{category}/{filename}.md`
- **Examples**:
  - `api_docs/document_slice_api.md`
  - `dev_docs/coding_standards.md`
  - `architecture/memory_system_design.md`

### Documentation Content Suggestions

A good document typically contains:

1. **Title and introduction**: Explain the purpose of the document
2. **Background/motivation**: Why this feature/change is needed
3. **Core content**: Detailed explanation
4. **Examples**: Code examples or usage examples
5. **Related documentation**: Links to other related documents

### Checklist

Before writing documentation, ask yourself:

- [ ] Is this change complex enough to need separate documentation?
- [ ] Are code comments already sufficient to explain the issue?
- [ ] Is the documentation in the correct directory?
- [ ] Is the documentation name clear and understandable?

---

## ⚡️ Async Programming Standards

### Full Async Architecture Principles

The project adopts **full async architecture**, based on the following principles:

#### 1. Single Event Loop Principle

- **The entire application uses one main Event Loop**
- Avoid creating new Event Loops in code (`asyncio.new_event_loop()`)
- Avoid using `asyncio.run()` to start new loops in async context

#### 2. About Using Threads and Processes ⚠️

**💡 Important Note: Be cautious with multithreading and multiprocessing**

The project is based on single Event Loop full async architecture, avoid the following operations:

```python
# ❌ Not recommended: Creating threads
import threading
thread = threading.Thread(target=some_function)
thread.start()

# ❌ Not recommended: Using thread pool (unless special cases)
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()

# ❌ Not recommended: Creating processes
import multiprocessing
process = multiprocessing.Process(target=some_function)
process.start()

# ❌ Not recommended: Using process pool
from concurrent.futures import ProcessPoolExecutor
executor = ProcessPoolExecutor()
```

**Why not recommended?**
- May break single Event Loop architecture, causing concurrency issues
- Thread safety issues are complex, easy to introduce race conditions
- Resource management is difficult, may cause resource leaks
- May affect async context (contextvars) normal operation
- Debugging becomes harder, stack traces are complex

**Special Case Handling**

If you really need to use threads or processes (e.g., CPU-intensive computation, calling third-party libraries that don't support async), it's recommended to:

1. **Discuss with development lead in advance**
2. Explain why async solution cannot meet the needs
3. Provide resource management plan (ensure threads/processes are properly closed)
4. Go through Code Review

**Allowed scenario examples**:

```python
# ✅ Special case: Calling sync libraries that don't support async (after discussion)
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Globally shared thread pool, limit max threads
_EXECUTOR = ThreadPoolExecutor(max_workers=4)

async def call_sync_library(data):
    """Call third-party library that doesn't support async (confirmed with lead)"""
    loop = asyncio.get_event_loop()
    # Run in thread pool to avoid blocking main loop
    result = await loop.run_in_executor(
        _EXECUTOR,
        sync_blocking_function,
        data
    )
    return result
```

#### 3. Async Function Definition

I/O operations should use async functions:

```python
# ✅ Correct: Async function
async def fetch_user_data(user_id: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"/users/{user_id}")
        return response.json()

# ❌ Wrong: Sync I/O
def fetch_user_data(user_id: str) -> dict:
    response = requests.get(f"/users/{user_id}")
    return response.json()
```

#### 4. Database Operations

```python
# ✅ Correct: Using async database driver
from pymongo import AsyncMongoClient

async def get_user(db, user_id: str):
    return await db.users.find_one({"_id": user_id})

# ❌ Wrong: Using sync driver
from pymongo import MongoClient

def get_user(db, user_id: str):
    return db.users.find_one({"_id": user_id})
```

#### 5. HTTP Client

```python
# ✅ Correct: Using httpx.AsyncClient
import httpx

async def call_api(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# ❌ Wrong: Using requests
import requests

def call_api(url: str):
    response = requests.get(url)
    return response.json()
```

#### 6. Concurrent Processing

Use `asyncio.gather()` for concurrent operations:

```python
# ✅ Correct: Execute multiple tasks concurrently
async def fetch_multiple_users(user_ids: list[str]):
    tasks = [fetch_user_data(uid) for uid in user_ids]
    results = await asyncio.gather(*tasks)
    return results

# ❌ Wrong: Serial execution
async def fetch_multiple_users(user_ids: list[str]):
    results = []
    for uid in user_ids:
        result = await fetch_user_data(uid)
        results.append(result)
    return results
```

#### 7. Prohibit I/O Operations in for Loops ⚠️

**💡 Important Note: Avoid serial I/O operations in loops**

Doing database access, API calls and other I/O operations in for loops causes serious performance issues, because each operation needs to wait for the previous one to complete, unable to take advantage of async concurrency.

**❌ Wrong example: I/O operations in loops**

```python
# Wrong: Serial database access in loop
async def get_users_info(user_ids: list[str]):
    results = []
    for user_id in user_ids:
        # Each loop iteration waits for database return, very poor performance
        user = await db.users.find_one({"_id": user_id})
        results.append(user)
    return results

# Wrong: Serial API calls in loop
async def fetch_user_profiles(user_ids: list[str]):
    profiles = []
    for user_id in user_ids:
        # Each loop iteration waits for API response, wasting time
        response = await api_client.get(f"/users/{user_id}")
        profiles.append(response.json())
    return profiles

# Wrong: Batch database inserts in loop
async def save_messages(messages: list[dict]):
    for msg in messages:
        # Each message inserted separately, very inefficient
        await db.messages.insert_one(msg)
```

**✅ Correct example: Using concurrent or batch operations**

```python
# Correct: Using asyncio.gather for concurrent execution
async def get_users_info(user_ids: list[str]):
    tasks = [db.users.find_one({"_id": uid}) for uid in user_ids]
    results = await asyncio.gather(*tasks)
    return results

# Correct: Using asyncio.gather for concurrent API calls
async def fetch_user_profiles(user_ids: list[str]):
    tasks = [api_client.get(f"/users/{uid}") for uid in user_ids]
    responses = await asyncio.gather(*tasks)
    return [r.json() for r in responses]

# Correct: Using batch insert operation
async def save_messages(messages: list[dict]):
    if messages:
        await db.messages.insert_many(messages)

# Correct: Using database's in query instead of loop query
async def get_users_info(user_ids: list[str]):
    # Single query to get all data
    cursor = db.users.find({"_id": {"$in": user_ids}})
    results = await cursor.to_list(length=None)
    return results
```

**Performance Comparison**

Assuming 100 users, each database query takes 10ms:
- ❌ Loop serial query: 100 × 10ms = 1000ms (1 second)
- ✅ Concurrent query: ~10ms (almost simultaneous completion)
- ✅ Batch query: ~10ms (single query)

**Exception Cases**

In rare cases you may need to do I/O in loops, but must meet the following conditions:

1. **Subsequent operations depend on previous result**: Must wait for previous operation to complete before next one
2. **Rate limiting needs**: Need to control concurrency to avoid pressure on external services
3. **Approved by development lead**

```python
# Allowed: Serial operations with dependencies (comment explaining reason)
async def process_workflow(steps: list[dict]):
    result = None
    for step in steps:
        # Each step depends on previous step's result, cannot be concurrent
        result = await execute_step(step, previous_result=result)
    return result

# Allowed: Using semaphore to control concurrency (comment explaining reason)
async def fetch_with_rate_limit(urls: list[str]):
    # Limit max 5 concurrent requests to avoid triggering external API rate limiting
    semaphore = asyncio.Semaphore(5)
    
    async def fetch_one(url: str):
        async with semaphore:
            return await api_client.get(url)
    
    tasks = [fetch_one(url) for url in urls]
    return await asyncio.gather(*tasks)
```

---

## 🕐 Timezone Awareness Standards

### Core Principle

**💡 Important Note: All time fields must be timezone-aware**

When handling date and time data, must ensure all time fields carry timezone information to avoid data errors and business issues caused by unclear timezone.

**⚠️ Prohibit direct use of `datetime` module standard methods**

The project uniformly uses utility functions from `common_utils/datetime_utils.py` for time handling, prohibit direct use of:
- ❌ `datetime.datetime.now()`
- ❌ `datetime.datetime.utcnow()`
- ❌ `datetime.datetime.today()`

Must use project-provided utility functions:
- ✅ `get_now_with_timezone()` - Get current time (with timezone)
- ✅ `from_timestamp()` - Convert from timestamp
- ✅ `from_iso_format()` - Convert from ISO format string
- ✅ `to_iso_format()` - Convert to ISO format string
- ✅ `to_timestamp()` / `to_timestamp_ms()` - Convert to timestamp

### Timezone Handling Rules

#### 1. Input Data Timezone Requirements

All time fields entering the system must meet:

- **Must carry timezone info**: All datetime type fields must be timezone-aware
- **Default timezone**: If input data doesn't have timezone info, treat it as the timezone configured by environment variable `TZ` (**default UTC**)
- **Storage format**: When storing in database, recommend converting to UTC timezone uniformly, but must preserve timezone info

#### 2. Python Implementation Standards

**✅ Correct example: Using project utility functions**

```python
from common_utils.datetime_utils import (
    get_now_with_timezone,
    from_timestamp,
    from_iso_format,
    to_iso_format,
    to_timestamp_ms,
    to_timezone
)

# Method 1: Get current time (automatically with timezone, configured by TZ env var, default UTC)
now = get_now_with_timezone()
# Returns: datetime.datetime(2025, 9, 16, 12, 17, 41, tzinfo=zoneinfo.ZoneInfo(key='UTC'))

# Method 2: Convert from timestamp (auto-detect seconds/milliseconds, auto-add timezone)
dt = from_timestamp(1758025061)
dt_ms = from_timestamp(1758025061000)

# Method 3: Convert from ISO string (auto-handle timezone)
dt = from_iso_format("2025-09-15T13:11:15.588000")  # No timezone, auto-add default timezone
dt_with_tz = from_iso_format("2025-09-15T13:11:15Z")  # Has timezone, preserve original then convert

# Method 4: Format to ISO string (auto-include timezone)
iso_str = to_iso_format(now)
# Returns: "2025-09-16T12:20:06.517301Z"

# Method 5: Convert to timestamp
ts = to_timestamp_ms(now)
# Returns: 1758025061123
```

**❌ Wrong example: Direct use of datetime module**

```python
import datetime

# ❌ Wrong: Prohibit using datetime.datetime.now()
naive_dt = datetime.datetime.now()  # Timezone unclear, prohibit!

# ❌ Wrong: Prohibit using datetime.datetime.utcnow()
dt = datetime.datetime.utcnow()  # Deprecated in Python 3.12+, prohibit!

# ❌ Wrong: Prohibit using datetime.datetime.today()
dt = datetime.datetime.today()  # Timezone unclear, prohibit!

# ❌ Wrong: Manually creating naive datetime
naive_dt = datetime.datetime(2025, 1, 1, 12, 0, 0)  # No timezone info
```

**🔧 How to fix existing code**

```python
# Old code (wrong)
import datetime
now = datetime.datetime.now()

# New code (correct)
from common_utils.datetime_utils import get_now_with_timezone
now = get_now_with_timezone()

# ----------------

# Old code (wrong)
from datetime import datetime
dt = datetime(2025, 1, 1, 12, 0, 0)

# New code (correct)
from common_utils.datetime_utils import from_iso_format
dt = from_iso_format("2025-01-01T12:00:00")  # Auto-add default timezone

# ----------------

# Old code (wrong)
ts = int(datetime.now().timestamp() * 1000)

# New code (correct)
from common_utils.datetime_utils import get_now_with_timezone, to_timestamp_ms
ts = to_timestamp_ms(get_now_with_timezone())
```

### Checklist

During code review, please confirm:

- [ ] **Prohibit direct use of `datetime.datetime.now()`**, must use `get_now_with_timezone()`
- [ ] **Prohibit direct use of `datetime.datetime.utcnow()`** or `datetime.datetime.today()`
- [ ] All time retrieval goes through utility functions in `common_utils/datetime_utils.py`
- [ ] Time parsed from external input uses `from_iso_format()` or `from_timestamp()`
- [ ] Time formatting uses `to_iso_format()` instead of manually calling `.isoformat()`
- [ ] Timestamp conversion uses `to_timestamp_ms()` instead of manual calculation
- [ ] Database schema uses timezone-aware types (e.g., `timestamptz`)
- [ ] API response time strings include timezone info (ISO 8601 format)
- [ ] Test data in unit tests all have timezone info

---

## 🏛️ Data Access Standards

### Core Principle

**💡 Important Note: All external storage access must go through infra layer repository**

When handling databases, search engines and other external storage systems, must follow strict layered architecture principles. All data read/write operations must be converged to `infra_layer` `repository` layer, prohibit direct calls to external storage capabilities in business layer or other upper layers.

**⚠️ Prohibit direct external storage access in these layers**
- ❌ `biz_layer` (Business layer)
- ❌ `memory_layer` (Memory layer)
- ❌ `agentic_layer` (Agent layer)
- ❌ API interface layer (`api_specs`)
- ❌ Application layer (`app.py`, controllers, etc.)

**✅ Must access through**
- `infra_layer/adapters/out/persistence/repository/` - Database access
- `infra_layer/adapters/out/search/repository/` - Search engine access

### Why This Standard?

#### 1. Separation of Concerns

Following Hexagonal Architecture and Clean Architecture principles:
- **Business layer**: Focus on business logic, don't care where data comes from
- **Infrastructure layer**: Handle all external system interaction details
- **Isolate changes**: When changing database or search engine, only need to modify infra layer

#### 2. Testability

```python
# ✅ Benefit: Business layer depends on abstract interface, easy to mock test
async def process_user_memory(user_id: str, memory_repo: MemoryRepository):
    """Business logic doesn't depend on specific implementation"""
    memories = await memory_repo.find_by_user_id(user_id)
    # Business processing...
    
# Can easily replace with mock during testing
mock_repo = MockMemoryRepository()
await process_user_memory("user_1", mock_repo)
```

#### 3. Code Reuse and Consistency

- Avoid repeatedly writing same database query logic in multiple places
- Unified exception handling, logging, performance monitoring
- Unified data transformation, validation

#### 4. Centralized Performance Optimization

- Index optimization, query optimization implemented uniformly in repository layer
- Cache strategies managed uniformly
- Batch operation optimization done in one place, benefits entire project

### Correct Architecture Layering

```
┌─────────────────────────────────────────┐
│  API Layer (api_specs, app.py)         │
│  - Receive requests, return responses   │
└─────────────┬───────────────────────────┘
              │ calls
              ▼
┌─────────────────────────────────────────┐
│  Business Layer (biz_layer)             │
│  - Business logic processing            │
│  - Depends on abstract interfaces (Port)│
└─────────────┬───────────────────────────┘
              │ dependency injection
              ▼
┌─────────────────────────────────────────┐
│  Memory Layer (memory_layer)            │
│  - Memory management logic              │
│  - Depends on abstract interfaces (Port)│
└─────────────┬───────────────────────────┘
              │ dependency injection
              ▼
┌─────────────────────────────────────────┐
│  Infrastructure Layer (infra_layer)     │
│  - Repository implementation (Adapter)  │
│  - Directly operate database/search     │
│  - MongoDB, PostgreSQL, ES, Milvus      │
└─────────────────────────────────────────┘
```

### Checklist

When writing or reviewing code, please confirm:

- [ ] **Are database operations in infra_layer/repository?**
- [ ] **Are search engine operations in infra_layer/repository?**
- [ ] **Does business layer depend on abstract interfaces (Port) not concrete implementations?**
- [ ] **Is dependency injection used to pass repository?**
- [ ] **Avoid directly creating database connections in business/API/application layers?**
- [ ] **Avoid directly using MongoDB/PostgreSQL/ES/Milvus clients in business layer?**
- [ ] **Has new Repository been registered in dependency injection container?**
- [ ] **Do Repository methods have clear business semantics (not exposing underlying implementation details)?**

---

## 📥 Import Standards

### PYTHONPATH Management

**💡 Important Note: PYTHONPATH needs unified management**

The project uniformly manages `PYTHONPATH` and module import paths. Changes involving path configuration should be discussed with development lead before unified configuration.

#### Why Unified Management?

- Chaotic import paths may cause modules not found or import errors
- Inconsistent paths across environments (dev/test/prod) may cause deployment issues
- Inconsistent IDE configuration may affect team collaboration
- Mixing relative and absolute imports increases code maintenance difficulty

#### Management Scope

The following directories in the project should maintain unified import paths:

- `src/`: Main business code
- `tests/`: Test code
- `unit_test/`: Unit tests
- `evaluation/`: Evaluation scripts
- Other directories needing to be imported (e.g., `demo/`)

#### Recommended Practices

1. **Unified project root directory**
   - Project root is `/Users/admin/memsys` (or corresponding deployment path)
   - src directory added to PYTHONPATH, import directly from module name

2. **Import standard examples**

```python
# ✅ Recommended: Absolute import (src already in PYTHONPATH)
from core.memory.manager import MemoryManager
from infra_layer.adapters.out.db import MongoDBAdapter
from tests.fixtures.mock_data import get_mock_user

# ✅ Recommended: Import in test files
from unit_test.email_data_constructor import construct_email

# ❌ Not recommended: Cross-level relative import
from ...core.memory.manager import MemoryManager

# ❌ Not recommended: Including src prefix (src already in PYTHONPATH, no prefix needed)
from src.core.memory.manager import MemoryManager

# ❌ Not recommended: Using sys.path.append to temporarily modify path
import sys
sys.path.append("../src")  # May cause environment inconsistency
```

### Prefer Absolute Imports

**💡 Important Note: Recommend absolute imports, avoid relative imports**

#### Why Recommend Absolute Imports?

Although relative imports are more concise in some scenarios, they have these issues:

- **Poor readability**: `from ...core.memory import Manager` is less intuitive than `from core.memory import Manager`
- **Difficult refactoring**: Moving files requires modifying all relative import levels
- **Complex debugging**: Stack traces with relative import paths are unclear
- **Tool support**: IDE and static analysis tools support absolute imports better
- **Testing convenience**: Test files using absolute imports are easier to understand dependencies

#### Import Style Comparison

```python
# ✅ Recommended: Absolute import (src already in PYTHONPATH)
from core.memory.manager import MemoryManager
from core.memory.types import MemoryType, MemoryStatus
from infra_layer.adapters.out.db.mongodb import MongoDBAdapter
from common_utils.logger import get_logger

# ✅ Acceptable: Single-level relative import within same package
# File: src/core/memory/manager.py
from .types import MemoryType  # Same directory
from .extractors.base import BaseExtractor  # Subdirectory

# ❌ Not recommended: Cross-level relative import
from ...infra_layer.adapters import MongoDBAdapter
from ....common_utils.logger import get_logger

# ❌ Not recommended: Multi-level upward relative import (hard to maintain)
from ......some_module import something
```

### __init__.py Usage Standards

**💡 Important Note: Not recommended to write any code in `__init__.py`**

#### Why Keep `__init__.py` Empty?

- **Import side effects**: `__init__.py` executes when package is imported, any code may produce unexpected side effects
- **Circular dependencies**: Even simple module exports easily cause circular import issues
- **Performance impact**: Code execution during import affects startup performance and module loading speed
- **Maintainability**: Code scattered in `__init__.py` is hard to locate and maintain
- **Testing difficulty**: Mock and unit tests become complex
- **Implicit behavior**: Implicit execution during import increases code understanding difficulty

#### Recommended Usage

**✅ Recommended: Keep as empty file**

```python
# src/core/memory/__init__.py

# Empty file, only serves as Python package identifier
# Do not write any code here
```

**How to import modules?**

Import directly from specific module files, don't rely on `__init__.py` re-export:

```python
# ✅ Recommended: Import directly from module files
from core.memory.manager import MemoryManager
from core.memory.types import MemoryType, MemoryStatus
from core.memory.extractors.base import BaseExtractor

# ❌ Not recommended: Relying on __init__.py re-export
from core.memory import MemoryManager  # Requires export code in __init__.py
```

#### Checklist

When writing or reviewing `__init__.py`, please confirm:

- [ ] File is empty (or only contains comments)?
- [ ] No import statements?
- [ ] No variable or constant definitions?
- [ ] No global object instances created?
- [ ] No classes or functions defined?
- [ ] No logic executed?

**If any of the above answers "no", move the code to a separate module file.**

---

## 📁 Module Introduction File Naming

### Core Principle

**💡 Important Note: Use `introduction.md` as module introduction file**

In subdirectories under `src/core/`, uniformly use lowercase `introduction.md` as module introduction file, not uppercase `README.md`.

### Why Not Use README.md?

- `README.md` may be auto-generated or legacy files
- Using `introduction.md` clearly distinguishes manually written module introductions from auto-generated content
- Maintains naming consistency and predictability

### Naming Examples

```
src/core/
├── di/
│   └── introduction.md              # DI module introduction
├── addons/
│   └── introduction.md              # Addons module introduction
├── component/
│   └── introduction.md              # Component module introduction
└── memory/
    └── introduction.md              # Memory module introduction
```

### introduction.md Content Suggestions

A good module introduction file should include:

1. **Module overview**: Module functionality and positioning
2. **Directory structure**: File organization within the module
3. **Core features**: Main classes, functions, and interface descriptions
4. **Usage examples**: Basic usage code examples
5. **Related documentation**: Links to other related documents

---

## 🌍 Internationalization Standards

### Core Principle

**💡 Important Note: Use English for code comments and documentation**

For international team communication and collaboration, all code comments and documentation should be written in English to ensure team members from different language backgrounds can understand and contribute to the project.

### Language Requirements

#### 1. Code Comments

All code comments must be written in English:

```python
# ✅ Recommended: English comments
def calculate_memory_score(memory: Memory, query: str) -> float:
    """
    Calculate relevance score between memory and query.
    
    Args:
        memory: The memory object to evaluate
        query: The search query string
    
    Returns:
        A float score between 0 and 1 indicating relevance
    """
    # Calculate base similarity score using cosine similarity
    base_score = cosine_similarity(memory.embedding, query_embedding)
    
    # Apply time decay factor: newer memories have higher weight
    time_decay = calculate_time_decay(memory.created_at)
    
    return base_score * time_decay

# ❌ Not recommended: Non-English comments
def calculate_memory_score(memory: Memory, query: str) -> float:
    """
    计算记忆与查询的相关性得分。
    """
    # 计算基础相似度得分
    base_score = cosine_similarity(memory.embedding, query_embedding)
    ...
```

#### 2. Documentation

All documentation files should be written in English:

- Technical documentation
- API documentation
- Module introduction files (`introduction.md`)
- Development guides
- Architecture documentation

**Note**: Chinese documentation can be maintained as supplementary reference (`*_zh.md`), but English should be the primary version.

#### 3. Commit Messages

Git commit messages should be written in English:

```bash
# ✅ Recommended: English commit messages
git commit -m "feat: add user authentication endpoint"
git commit -m "fix: resolve memory leak in cache manager"
git commit -m "docs: update API documentation for search endpoint"

# ❌ Not recommended: Non-English commit messages
git commit -m "功能: 添加用户认证接口"
```

#### 4. Code Naming

Variable names, function names, class names, and other identifiers should use English:

```python
# ✅ Recommended: English naming
class MemoryManager:
    def __init__(self):
        self.cache_size = 1000
        self.max_retries = 3
    
    async def fetch_user_memories(self, user_id: str) -> list[Memory]:
        ...

# ❌ Not recommended: Non-English or mixed naming
class JiyiGuanliqi:  # Pinyin naming
    def __init__(self):
        self.huancun_daxiao = 1000  # Pinyin variable name
```

### Benefits

- **Global Collaboration**: Enables team members worldwide to understand and contribute
- **Industry Standard**: Aligns with international software development practices
- **Tool Compatibility**: Better support from IDEs, linters, and documentation generators
- **Knowledge Sharing**: Easier to share code and documentation with the open-source community

### Checklist

Before submitting code, confirm:

- [ ] All code comments are written in English
- [ ] Docstrings are written in English
- [ ] Commit messages are written in English
- [ ] Variable and function names use English
- [ ] New documentation is written in English (with optional Chinese supplement)

---

## 🌿 Branch Management Standards

### Branch Type Descriptions

| Branch | Description | Notes |
|--------|-------------|-------|
| `master` | Stable version; only bug fix branches cut from here, `release/xxx` and `hotfix/xxx` merge here | Production deployment branch |
| `dev` | Daily development version; continuous code commits | If versioning has started & commit is for this version, commit to `release`; non-urgent small bugs & features merge to `dev`, catch next release |
| `release/YYMMDD` | Versioning branch; first deploy to test, then production; first merge `dev` to `master`, then cut from `dev`; after actual release merge back to `master`, `dev` | Currently irregular (notified in group); only this release's bug or code commits |
| `feature/xxxx` | Single cycle, small feature; merge to `dev` or some `release` | Merge to `dev` can be direct; merge to `release` recommend MR |
| `bugfix/xxxx` | Single cycle, small bug; merge to `dev` or some `release` | Merge to `dev` can be direct; merge to `release` recommend MR |
| `long/xxx` | Cross-cycle, large feature; cut from `dev`, merge to `dev` or some `release` | Separate test in new test environment (port/address distinction); regularly merge `dev` to avoid too many conflicts at end; recommend MR |
| `hotfix/xxxx` | Bug fix; cut from `master`, MR to `master` branch (`dev` if needed) | Only exists after release; normal dev stage bugs merge directly on `dev`, during versioning but before release merge to `release`, urgent bugs without current versioning use this branch; recommend MR |

### Environment and Branch Mapping

| Environment | Possible Branches | Notes |
|-------------|------------------|-------|
| Production | `master` branch | Stable version |
|            | `release/xxx` branch | After versioned release and before bug fix |
| Testing | `dev` branch | Daily development stage |
|         | `release/xxx` branch | Versioning test stage |
|         | `hotfix/xxxx` | Emergency bug fix |

### Version Tag Standards

| Tag | Description | Notes |
|-----|-------------|-------|
| `X.Y.Z` | Version number: Major.Iteration.BugFix | May not sync with iterations, add when needed |

- **X (Major version)**: Major architecture changes or incompatible updates
- **Y (Iteration version)**: Feature iterations, new features added
- **Z (Fix version)**: Bug fixes, small optimizations

### Branch Operation Flows

#### 1. Daily Development (feature/bugfix)

```bash
# Create feature branch from dev
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name

# After development complete
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature-name

# Merge to dev (small features can merge directly)
git checkout dev
git merge feature/your-feature-name
git push origin dev

# Delete feature branch
git branch -d feature/your-feature-name
git push origin --delete feature/your-feature-name
```

#### 2. Release Flow (release)

```bash
# 1. First merge dev to master (ensure includes latest hotfix)
git checkout dev
git pull origin dev
git merge master
git push origin dev

# 2. Create release branch from dev
git checkout -b release/$(date +%y%m%d)
git push origin release/$(date +%y%m%d)

# 3. Bug fixes during testing stage
git checkout release/$(date +%y%m%d)
# ... fix bugs ...
git commit -m "fix: bug description"
git push origin release/$(date +%y%m%d)

# 4. After release merge back to master and dev
git checkout master
git merge release/$(date +%y%m%d)
git tag -a v1.2.3 -m "Release version 1.2.3"
git push origin master --tags

git checkout dev
git merge release/$(date +%y%m%d)
git push origin dev
```

#### 3. Emergency Fix (hotfix)

```bash
# Create hotfix branch from master
git checkout master
git pull origin master
git checkout -b hotfix/critical-bug-fix

# After fix complete, recommend MR process
git add .
git commit -m "hotfix: critical bug description"
git push origin hotfix/critical-bug-fix

# Create Merge Request to master
# After merge remember to sync to dev
git checkout dev
git merge master
git push origin dev
```

#### 4. Long-term Feature Development (long)

```bash
# Create long branch from dev
git checkout dev
git pull origin dev
git checkout -b long/big-feature

# Regularly merge dev to avoid conflict accumulation
git checkout long/big-feature
git merge dev

# After feature complete, recommend MR process to merge to dev or release
```

### Unified Branch Merge Handling Standards

**⚠️ Important Note: The following branch merge operations need to be handled uniformly by development or operations lead**

To ensure code quality and release process standards, the following branch merge operations need to be managed and executed uniformly by development lead or operations lead:

#### Merge Operations Needing Unified Handling

1. **Long-term feature branch merge to dev**
   - `long/xxx` → `dev`
   - Reason: Long-term feature branches usually involve large code changes, need to evaluate impact scope and potential conflicts

2. **Cut release branch from dev**
   - `dev` → `release/YYMMDD`
   - Reason: Release nodes need unified coordination, ensure version content is complete and meets release requirements

3. **Merge release back to dev**
   - `release/YYMMDD` → `dev`
   - Reason: Ensure release branch bug fixes correctly sync back to main development branch

#### Notes

- Small feature branches (`feature/xxx`, `bugfix/xxx`) merging to `dev` can be done by developers
- Emergency `hotfix` merging to `master` recommend MR process with lead review
- All merges involving `release` and `master` recommend lead confirmation

---

## 📤 MR Standards

### Core Principles

#### 1. Small Steps, Reduce Single Commit Size

**💡 Important Note: Keep code commits small, iterate quickly, avoid submitting too much code at once**

Each MR should stay small and focused, easy to review and track issues.

**Why small steps?**

- **Easy to Review**: Smaller changes are easier to understand and review, higher Review quality
- **Fast feedback**: Small batch commits get feedback faster, adjust direction in time
- **Issue location**: When problems occur, easier to locate specific commit
- **Lower risk**: Risk of merging large amounts of code at once is much higher than multiple small merges
- **Reduce conflicts**: Frequent small batch merges reduce code conflict probability and complexity

**Recommended practices**:

```bash
# ✅ Recommended: Split commits by feature points or logical units
git commit -m "feat: add user authentication endpoint"
git commit -m "feat: add user authentication middleware"
git commit -m "test: add user authentication unit tests"

# ❌ Not recommended: Submit large amounts of unrelated changes at once
git commit -m "feat: complete all user module features"  # Contains dozens of file changes
```

**Commit split suggestions**:

| Commit Type | Recommended Size | Description |
|-------------|-----------------|-------------|
| **Feature development** | 50-200 lines | One independent feature point or logical unit |
| **Bug fix** | As small as possible | Only include code necessary for fix |
| **Refactoring** | 100-300 lines | Only one type of refactoring at a time |
| **Documentation** | Flexible | Documentation updates can be relatively flexible |

#### 2. Ensure Each Commit is Runnable

**💡 Important Note: Try not to submit broken or work-in-progress code, each commit should be runnable**

Each commit to shared branches (like `dev`, `release`) should be a runnable complete state.

**Why ensure commit quality?**

- **Continuous integration**: Ensure CI/CD pipeline won't fail due to incomplete code
- **Team collaboration**: Other developers can run and develop normally after pulling code
- **Fast rollback**: Any commit is a stable point that can be safely rolled back to
- **Code tracing**: Tools like `git bisect` need each commit to be runnable

**Pre-commit checklist**:

- [ ] Code passes pre-commit checks (formatting, lint, etc.)
- [ ] No obvious syntax or runtime errors
- [ ] Related unit tests pass
- [ ] Feature is complete, not half-finished
- [ ] No debug code (like `print` debug statements, commented out code blocks)
- [ ] No sensitive information (passwords, keys, tokens, etc.)

#### 3. Files Requiring Code Review

**💡 Important Note: The following types of file changes must go through Code Review**

To ensure code quality and system stability, changes to the following files or directories must create MR and assign reviewers:

##### Data-related Files

| File/Directory | Description | Risk Level |
|----------------|-------------|------------|
| `migrations/` | Database migration scripts | 🔴 High |
| `devops_scripts/data_fix/` | Data fix scripts | 🔴 High |
| Any batch scripts involving `insert`/`update`/`delete` | Batch data changes | 🔴 High |

##### Dependency-related Files

| File/Directory | Description | Risk Level |
|----------------|-------------|------------|
| `pyproject.toml` | Dependency configuration changes | 🟠 Medium-High |
| `uv.lock` | Dependency lock file changes | 🟠 Medium-High |

##### Infrastructure-related Files

| File/Directory | Description | Risk Level |
|----------------|-------------|------------|
| `infra_layer/` | Infrastructure layer code | 🟠 Medium-High |
| `bootstrap.py` | Application startup entry | 🔴 High |
| `application_startup.py` | Application startup flow | 🔴 High |
| `base_app.py` | Base application class | 🔴 High |
| Dependency injection container config | DI container configuration | 🟠 Medium-High |

##### Branch Merge Operations

| Operation Type | Description | Risk Level |
|----------------|-------------|------------|
| Merge to `release/xxx` | Release branch merge | 🟠 Medium-High |
| Merge to `master` | Main branch merge | 🔴 High |
| `long/xxx` → `dev` | Long-term branch merge | 🟠 Medium-High |

---

## 🔍 Code Review Process

### Data Migration and Schema Change Process

**⚠️ Important Principle: Plan ahead, communicate fully**

When launching new features involving data fixes or Schema migration, discuss feasibility and subsequent implementation timing with development lead and operations lead as early as possible.

#### Why Early Communication?

Data migration and Schema changes are high-risk operations that may affect:

- **Data integrity**: Data structure changes may cause data loss or corruption
- **Service availability**: Large-scale data migration may affect service performance
- **Rollback complexity**: Rollback after Schema changes is often more complex than code rollback
- **Time window**: Need sufficient time for data migration and verification
- **Multi-team collaboration**: Involves development, testing, operations multiple teams

#### Common Scenario Examples

| Scenario | Early Communication Time | Key Discussion Points |
|----------|-------------------------|----------------------|
| **Add new field** | Early development (1-2 weeks) | Default value strategy, index creation, whether to backfill historical data |
| **Field type change** | Design phase (2-3 weeks) | Data conversion rules, incompatible data handling, rollback plan |
| **Large-scale data fix** | Design phase (2-4 weeks) | Data volume estimation, migration duration, batch strategy, downtime plan |
| **Index rebuild** | Design phase (1-2 weeks) | Performance impact, execution time window, online/offline approach |
| **Data archiving/cleanup** | Design phase (2-3 weeks) | Archiving strategy, data backup, recovery mechanism |

### Code Review Flow

#### Submitter Recommendations

1. **Create Merge Request**
   - Fill in clear title and description
   - Explain change reasons and impact scope
   - Link related Issues or requirements

2. **Self-check list**
   - [ ] Code passes pre-commit checks
   - [ ] Related unit tests added/updated
   - [ ] Documentation updated (if necessary)
   - [ ] No obvious performance issues
   - [ ] No security risks

**Note**: Project has Code Owner mechanism configured, reviewers will be auto-assigned based on changed files, no need to manually specify.

#### Reviewer Work

1. **Code quality review**
   - Code logic correctness
   - Code readability and maintainability
   - Whether follows project standards

2. **Risk assessment**
   - Data security risks (especially for data scripts)
   - Performance impact (async code, database queries)
   - Compatibility issues (dependency upgrades, API changes)

3. **Review feedback**
   - Provide clear modification suggestions
   - Mark severity (Must Fix / Should Fix / Nice to Have)
   - Respond promptly (try within 24 hours)

---

## 📚 Related Documentation

- [Getting Started Guide](./getting_started.md)
- [Development Guide](./development_guide.md)
- [Dependency Management Guide](./project_deps_manage.md)
- [Bootstrap Usage Guide](./bootstrap_usage.md)
- [MongoDB Migration Guide](./mongodb_migration_guide.md)

---

## ❓ FAQ

### Q1: Forgot to install pre-commit hook?

```bash
make setup-hooks
pre-commit run --all-files  # Run check on existing code
```

### Q2: Accidentally installed package with pip?

```bash
# 1. Uninstall package installed with pip
pip uninstall <package-name>

# 2. Reinstall with uv
uv add <package-name>

# 3. Re-sync environment
uv sync
```

### Q3: Branch merge conflict?

```bash
# 1. Ensure local branch is up to date
git checkout your-branch
git pull origin your-branch

# 2. Merge target branch
git merge target-branch

# 3. After resolving conflicts, commit
git add .
git commit -m "merge: resolve conflicts with target-branch"
```

### Q4: How to call sync library in async code?

```python
import asyncio

async def use_sync_library():
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,  # Use default thread pool
        sync_function,
        arg1,
        arg2
    )
    return result
```

---

## 👤 Contacts

### Development Lead

For the following matters, recommend communicating with development lead:

- Thread/process usage plan discussion
- PYTHONPATH path configuration changes
- Code Review review requests
- Technical plans for data scripts, dependency changes, infrastructure changes

**Current lead**: zhanghui

### Operations Lead

For the following matters, contact operations lead:

- Development environment configuration (database, middleware connection info)
- Service access permission requests
- Environment configuration troubleshooting
- New configuration items or environment needs
- Network connection, VPN and other infrastructure issues

**Current lead**: jianhua

---

**Last updated**: 2025-10-31
