.PHONY: help install install-deps lint docs-check check-commits check-assets check-cjk check-datetime openapi check-openapi format test integration package cov ci clean

help:
	@echo "Targets:"
	@echo "  install       Install deps + pre-commit hooks (full dev setup)"
	@echo "  install-deps  Install deps only (uv sync --frozen, used by CI)"
	@echo "  lint          ruff + import-linter + repo asset/media + datetime discipline + openapi drift"
	@echo "  docs-check    Validate Markdown links, use-case banners, and issue template YAML"
	@echo "  check-commits Validate Conventional Commit subjects for a git range"
	@echo "  check-assets  Block committed images, videos, and asset/media directories"
	@echo "  check-cjk     Scan for CJK outside the language-policy allowlist (advisory)"
	@echo "  check-datetime Scan for code that bypasses component/utils/datetime (HARD gate, run via lint)"
	@echo "  openapi       Regenerate docs/openapi.json from the FastAPI app"
	@echo "  check-openapi Verify docs/openapi.json matches app.openapi() (HARD gate, run via lint)"
	@echo "  format        Format src/tests with ruff"
	@echo "  test          pytest tests/unit"
	@echo "  integration   pytest tests/integration"
	@echo "  package       Build sdist/wheel and smoke-test wheel import"
	@echo "  cov           pytest tests/unit + tests/integration with coverage (fail under 80%)"
	@echo "  ci            full CI: lint + test + integration + package"
	@echo "  clean         Remove caches"

# Sync deps from uv.lock; CI calls this directly. --frozen means "lock is the
# source of truth — fail rather than update it".
install-deps:
	uv sync --frozen

# One-stop dev setup: deps + pre-commit hooks (both pre-commit and commit-msg
# stages — gitlint runs on commit-msg).
install: install-deps
	uv run pre-commit install
	uv run pre-commit install --hook-type commit-msg

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
	uv run lint-imports
	uv run python scripts/check_repo_assets.py
	uv run python scripts/check_datetime_discipline.py
	uv run python scripts/dump_openapi.py --check

docs-check:
	python3 scripts/check_docs.py
	ruby -e 'require "yaml"; Dir[".github/ISSUE_TEMPLATE/*.yml"].sort.each { |p| YAML.load_file(p); puts "YAML ok: #{p}" }'

check-commits:
	python3 scripts/check_commit_messages.py $(RANGE)

# Repository media hygiene gate. Images/videos belong in external hosting,
# release artifacts, or other approved storage, then linked from docs.
check-assets:
	uv run python scripts/check_repo_assets.py

# Advisory CJK scan (see .claude/rules/language-policy.md). Deliberately NOT
# wired into `lint` / `ci`: the policy is enforced by review and the rules
# doc, not a hard gate. Run on demand when touching potentially-CJK files.
check-cjk:
	uv run python scripts/check_cjk.py

# Datetime two-zone discipline scanner (see .claude/rules/datetime-handling.md).
# Wired into `lint` (and therefore `ci`) as a HARD gate — any code that
# bypasses ``component/utils/datetime`` (raw ``datetime.now()``,
# ``time.time()``, naked ``datetime(...)`` constructor, etc.) fails the build.
check-datetime:
	uv run python scripts/check_datetime_discipline.py

# OpenAPI schema export — produce docs/openapi.json from the FastAPI app.
# Run this after touching any HTTP route / DTO; commit the result.
openapi:
	uv run python scripts/dump_openapi.py

# OpenAPI drift gate (wired into `lint`). Re-renders the schema in memory
# and diffs it against the committed ``docs/openapi.json``; any drift
# fails the build with a unified diff. Forces the contract doc to track
# the code on every PR that touches the API surface.
check-openapi:
	uv run python scripts/dump_openapi.py --check

format:
	uv run ruff check --fix src tests
	uv run ruff format src tests

test:
	uv run pytest tests/unit -v

integration:
	uv run pytest tests/integration -v

package:
	rm -rf dist .package-smoke
	uv build --sdist --wheel
	uv venv --python 3.12 --seed .package-smoke
	uv pip install --python .package-smoke/bin/python --no-deps dist/*.whl
	.package-smoke/bin/python -c "import everos; print(everos.__version__)"
	rm -rf .package-smoke

# Coverage runs unit + integration so the number matches what CI's `test` and
# `integration` jobs actually exercise. Threshold starts at 80% (unit-only is
# currently 87%, unit+integration 91% — 80% leaves ~10pp headroom for normal
# churn). Bump as the suite stabilises.
cov:
	uv run pytest tests/unit tests/integration --cov=src/everos --cov-report=term-missing --cov-branch --cov-fail-under=80

ci: lint test integration package

clean:
	rm -rf .pytest_cache .ruff_cache .uv-cache .mypy_cache .coverage htmlcov .package-smoke
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
