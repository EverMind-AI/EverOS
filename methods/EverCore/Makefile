.PHONY: dev-setup setup-hooks lint test clean help

# Default target
help:
	@echo "Available targets:"
	@echo "  dev-setup        - Full dev environment setup (sync deps + install hooks)"
	@echo "  setup-hooks      - Install pre-commit hooks only"
	@echo "  lint             - Run linters"
	@echo "  test             - Run tests"
	@echo "  clean            - Clean up generated files"
	@echo "  help             - Show this help message"

# Full development environment setup
dev-setup:
	@echo "Setting up development environment..."
	@echo "1. Syncing dependencies..."
	uv sync --dev
	@echo "2. Installing pre-commit hooks..."
	@$(MAKE) setup-hooks
	@echo "Development environment is ready!"

# Install pre-commit hooks (clean install)
setup-hooks:
	@echo "Removing existing hooks..."
	rm -f .git/hooks/pre-commit .git/hooks/commit-msg
	@echo "Installing pre-commit hooks..."
	pre-commit install --hook-type pre-commit --hook-type commit-msg
	@echo "Done! Pre-commit hooks installed."

# Run linters
lint:
	@echo "Running black..."
	black src/
	@echo "Running i18n check..."
	PYTHONPATH=src python -m devops_scripts.i18n.i18n_tool check

# Run tests
test:
	PYTHONPATH=src pytest tests/

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf .pytest_cache 2>/dev/null || true
	@echo "Cleaned up generated files."
