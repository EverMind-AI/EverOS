# CLAUDE.md

Refer to [AGENTS.md](methods/EverCore/AGENTS.md) for comprehensive project documentation including:
- Project architecture and structure
- Tech stack and dependencies
- Code conventions and patterns
- Key abstractions and files
- Development guidelines
- Database schema

## Quick Commands

```bash
cd methods/EverCore
docker-compose up -d          # Start infrastructure
uv sync                       # Install dependencies
make run                      # Run application
pytest                        # Run tests
black src/ && isort src/      # Format code
pyright                       # Type check
```

## Key Entry Points

- `methods/EverCore/src/run.py` - Application entry
- `methods/EverCore/src/agentic_layer/memory_manager.py` - Core memory manager
- `methods/EverCore/src/infra_layer/adapters/input/api/` - REST API controllers

## Remember

- All I/O is async - use `await`
- Multi-tenant system - data is tenant-scoped
- Prompts in `methods/EverCore/src/memory_layer/prompts/` (EN/ZH)
