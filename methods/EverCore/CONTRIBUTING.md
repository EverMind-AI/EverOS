# Contributing to EverOS

Thank you for your interest in contributing to EverOS! We welcome contributions from the community.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- `uv` package manager

### Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/EverOS.git
cd EverOS
```

2. Install dependencies:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp env.template .env
# Edit .env with your configuration
```

4. Start development services:
```bash
docker-compose up -d
```

## 📝 Code Style

### Python Guidelines

- Follow [PEP 8](https://pep8.org/) style guide
- Use type hints for function parameters and return values
- Add docstrings for classes and functions
- Maximum line length: 100 characters

### Key Rules

- **No relative imports**: Use absolute imports from project root
- **No wildcard imports**: Avoid `from module import *`
- **DateTime handling**: Use `common_utils.datetime_utils` instead of direct `datetime` module
- **No code in `__init__.py`**: Use only as package markers

## 🔀 Git Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

### Commit Messages

We use [Gitmoji](https://gitmoji.dev/) for commit messages.

**Format**: `<emoji> <type>: <description>`

**Examples**:
```
✨ feat: Add new memory retrieval algorithm
🐛 fix: Fix memory leak in vector indexing
📝 docs: Update API documentation
♻️ refactor: Simplify memory extraction logic
✅ test: Add tests for profile extraction
⚡ perf: Optimize vector search performance
```

**Common Gitmoji**:

| Emoji | Code | Usage |
|-------|------|-------|
| ✨ | `:sparkles:` | New feature |
| 🐛 | `:bug:` | Bug fix |
| 🚑 | `:ambulance:` | Critical hotfix |
| 📝 | `:memo:` | Documentation |
| ♻️ | `:recycle:` | Refactor code |
| 🔥 | `:fire:` | Remove code/files |
| ✅ | `:white_check_mark:` | Add tests |
| ⚡ | `:zap:` | Performance improvement |
| 🔧 | `:wrench:` | Configuration changes |
| 🗃️ | `:card_file_box:` | Database changes |
| ⬆️ | `:arrow_up:` | Upgrade dependencies |
| 🐳 | `:whale:` | Docker related |
| 🚀 | `:rocket:` | Deployment |

See [gitmoji.dev](https://gitmoji.dev/) for full reference.

### Pull Request Process

1. **Create a feature branch** from `main`:
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** following the code style guidelines

3. **Test your changes**:
```bash
# Run tests (if applicable)
pytest tests/

# Check code style
ruff check .
```

4. **Commit your changes** with clear, descriptive commit messages

5. **Push to your fork**:
```bash
git push origin feature/your-feature-name
```

6. **Open a Pull Request** with:
   - Clear description of changes
   - Reference to related issues (if any)
   - Screenshots (if UI changes)

7. **Address review feedback** promptly

## 🧪 Testing

- Add tests for new features
- Ensure existing tests pass
- Maintain or improve code coverage

## 📚 Documentation

- Update relevant documentation when changing functionality
- Add docstrings to new functions and classes
- Update README.md if adding major features
- Keep API documentation in sync with code changes

## 🐛 Reporting Bugs

Please report bugs by [creating a bug report](https://github.com/EverMind-AI/EverOS/issues/new?template=bug_report.md) with:

- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

## 💡 Suggesting Features

Feature requests are welcome! Please [submit a feature request](https://github.com/EverMind-AI/EverOS/issues/new?template=feature_request.md) with:

- Check if the feature is already requested
- Provide clear use cases
- Explain why this feature would be useful
- Consider backward compatibility

## 📄 License

By contributing to EverOS, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

## 🤝 Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Maintain a professional environment

## 📞 Questions?

- Open a [Discussion](https://github.com/EverMind-AI/EverOS/discussions)
- Join our community channels
- Email: evermind@shanda.com

---

Thank you for contributing to EverOS! 🎉

