# Architecture Design

[Home](../README.md) > [Docs](README.md) > Architecture

## Overview

EverMemOS adopts a layered architecture design that separates concerns and enables scalability. The system is built around two main cognitive tracks: **memory construction** and **memory perception**, which work together to create a comprehensive AI memory system.

For a high-level overview of the system framework, see [System Framework in Overview](OVERVIEW.md#system-framework).

---

## Layered Architecture

EverMemOS is organized into six main layers, each with specific responsibilities:

### 1. Agentic Layer

The top-level orchestration layer that provides a unified memory interface.

- **Responsibilities:**
  - Memory extraction coordination
  - Vectorization management
  - Retrieval orchestration
  - Reranking operations

- **Key Components:**
  - Memory extraction agents
  - Vector embedding services
  - Retrieval coordinators
  - Reranking engines

### 2. Memory Layer

Handles memory extraction and management.

- **Responsibilities:**
  - MemCell extraction from conversations
  - Episodic memory management
  - Memory type classification (episodes, profiles, preferences, etc.)
  - LLM prompt management for extraction

- **Key Components:**
  - `memcell_extractor/` - Atomic memory unit extraction
  - `memory_extractor/` - Higher-level memory construction
  - `prompts/` - LLM prompt templates

### 3. Retrieval Layer

Implements multi-modal retrieval and result ranking.

- **Responsibilities:**
  - Semantic search (vector-based)
  - Keyword search (BM25)
  - Hybrid retrieval (RRF fusion)
  - Agentic multi-round recall
  - Result reranking

- **Key Components:**
  - Vector search engines (Milvus integration)
  - Keyword search engines (Elasticsearch integration)
  - Fusion algorithms
  - Reranking services

### 4. Business Layer

Contains business logic and data operations.

- **Responsibilities:**
  - API endpoint implementations
  - Request/response handling
  - Business rule enforcement
  - Data validation

- **Key Components:**
  - Controllers
  - Service layer
  - Data transformation logic

### 5. Infrastructure Layer

Provides adapters for external services and databases.

- **Responsibilities:**
  - Database connections (MongoDB)
  - Cache management (Redis)
  - Search engine integration (Elasticsearch, Milvus)
  - Message queue handling

- **Key Components:**
  - Database adapters
  - Cache adapters
  - Search engine clients
  - Queue managers

### 6. Core Framework

Foundation layer providing cross-cutting concerns.

- **Responsibilities:**
  - Dependency injection
  - Lifecycle management
  - Middleware pipeline
  - Queue management
  - Configuration management

- **Key Components:**
  - DI container (see [DI Framework](../src/core/di/README.md))
  - Lifecycle hooks
  - Middleware system
  - Common utilities

---

## Project Structure

```
evermemos-opensource/
├── src/                              # Source code directory
│   ├── agentic_layer/                # Agentic layer - unified memory interface
│   ├── memory_layer/                 # Memory layer - memory extraction
│   │   ├── memcell_extractor/        # MemCell extractor
│   │   ├── memory_extractor/         # Memory extractor
│   │   └── prompts/                  # LLM prompt templates
│   ├── retrieval_layer/              # Retrieval layer - memory retrieval
│   ├── biz_layer/                    # Business layer - business logic
│   ├── infra_layer/                  # Infrastructure layer
│   ├── core/                         # Core functionality (DI/lifecycle/middleware)
│   ├── component/                    # Components (LLM adapters, etc.)
│   └── common_utils/                 # Common utilities
├── demo/                             # Demo code
├── data/                             # Sample conversation data
├── evaluation/                       # Evaluation scripts
│   └── src/                          # Evaluation framework source code
├── data_format/                      # Data format definitions
├── docs/                             # Documentation
├── config.json                       # Configuration file
├── env.template                      # Environment variable template
├── pyproject.toml                    # Project configuration
└── README.md                         # Project description
```

---

## Technology Stack

### Core Technologies

- **FastAPI** - Modern web framework for building APIs
- **Python 3.10+** - Primary programming language
- **uv** - Fast Python package manager

### Storage & Search

- **MongoDB 7.0+** - Primary database for memory cells and profiles
- **Elasticsearch 8.x** - Keyword search engine (BM25)
- **Milvus 2.4+** - Vector database for semantic retrieval
- **Redis 7.x** - Cache service for performance optimization

### AI/ML Services

- **LLM APIs** - For memory extraction and reasoning
- **Embedding Models** - For semantic vectorization
- **Reranker Models** - For relevance scoring

### Infrastructure

- **Docker & Docker Compose** - Containerization and orchestration
- **Beanie** - Async ODM for MongoDB

---

## Data Flow

### Memory Construction Flow

```
User Conversation
    ↓
Message Ingestion (API)
    ↓
MemCell Extraction (Memory Layer)
    ↓
Memory Type Classification
    ↓
Storage (MongoDB)
    ↓
Indexing (Elasticsearch + Milvus)
```

### Memory Retrieval Flow

```
User Query
    ↓
Retrieval Mode Selection
    ↓
├─ Lightweight Mode
│   ├─ BM25 Search (Elasticsearch)
│   ├─ Vector Search (Milvus)
│   └─ RRF Fusion
│
└─ Agentic Mode
    ├─ Query Expansion (LLM)
    ├─ Multi-round Retrieval
    └─ Intelligent Fusion
    ↓
Reranking (Optional)
    ↓
Results to User
```

---

## Design Principles

### 1. Separation of Concerns

Each layer has a well-defined responsibility, making the system easier to understand, test, and maintain.

### 2. Scalability

The layered architecture allows individual components to scale independently based on load.

### 3. Flexibility

Multiple retrieval strategies (lightweight vs agentic) allow users to choose based on their latency and accuracy requirements.

### 4. Extensibility

New memory types, retrieval strategies, or storage backends can be added without major refactoring.

### 5. Testability

Clear layer boundaries enable unit testing of individual components and integration testing of layer interactions.

---

## Memory Construction Architecture

### MemCell: Atomic Memory Unit

MemCells are the fundamental building blocks of the memory system. Each MemCell represents a single, atomic piece of information extracted from a conversation.

**MemCell Properties:**
- Unique identifier
- Content (the extracted information)
- Metadata (timestamp, participants, etc.)
- Memory type classification
- Semantic embeddings

### Memory Types

EverMemOS supports multiple memory types, each serving different purposes:

- **Episodes** - Coherent conversation threads on specific topics
- **Profiles** - User characteristics and attributes
- **Preferences** - User likes, dislikes, and choices
- **Relationships** - Connections between people
- **Semantic Knowledge** - Facts and information
- **Basic Facts** - Simple factual statements
- **Core Memories** - Important, long-lasting memories

See [Memory Types Guide](dev_docs/memory_types_guide.md) for detailed information.

---

## Memory Perception Architecture

### Retrieval Strategies

#### Lightweight Retrieval

Fast, efficient retrieval for latency-sensitive scenarios:
- **BM25** - Keyword-based search
- **Embedding** - Semantic vector search
- **RRF** - Hybrid fusion of both methods

#### Agentic Retrieval

Intelligent, multi-round retrieval for complex queries:
- Query expansion using LLM
- Multiple retrieval paths
- Intelligent result fusion

See [Retrieval Strategies Guide](advanced/RETRIEVAL_STRATEGIES.md) for more details.

### Reranking

Optional reranking step to improve result relevance:
- Deep relevance scoring
- Batch processing with retry logic
- Prioritization of critical information

---

## Configuration Management

### Environment Variables

Key configuration is managed through environment variables (see `.env`):
- LLM API credentials
- Embedding service credentials
- Database connection strings
- Service endpoints

See [Configuration Guide](usage/CONFIGURATION_GUIDE.md) for complete details.

### Service Configuration

Each service (MongoDB, Elasticsearch, Milvus, Redis) can be configured independently for:
- Resource allocation
- Performance tuning
- Network settings
- Security settings

See [Docker Setup Guide](installation/DOCKER_SETUP.md) for service-specific configuration.

---

## For Developers

If you're contributing to EverMemOS, these resources will help:

- **[Development Guide](dev_docs/development_guide.md)** - Architecture details and best practices
- **[Development Standards](dev_docs/development_standards.md)** - Code standards and conventions
- **[DI Framework](../src/core/di/README.md)** - Understanding the dependency injection system
- **[Contributing Guide](../CONTRIBUTING.md)** - How to contribute code

---

## See Also

- [Overview & System Framework](OVERVIEW.md)
- [Memory Types Guide](dev_docs/memory_types_guide.md)
- [Development Guide](dev_docs/development_guide.md)
- [API Documentation](api_docs/memory_api.md)
