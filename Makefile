.PHONY: sync test test-cov test-ci cov-report cov lint-typing lint-style lint-fmt lint-check lint-typos lint-all security-bandit security-audit security baml-gen clean help

help:
	@echo "Available make targets:"
	@echo "  make sync             - Sync project and install dependencies"
	@echo "  make test             - Run unit tests"
	@echo "  make test-integration - Run integration tests (requires network)"
	@echo "  make test-cov         - Run tests with coverage collection"
	@echo "  make test-ci          - Run tests with coverage + XML/junit output for CI"
	@echo "  make cov-report       - Generate coverage reports (xml, html)"
	@echo "  make cov              - Run tests and generate coverage reports"
	@echo "  make lint-typing      - Type check with ty"
	@echo "  make lint-style       - Lint with ruff (check only)"
	@echo "  make lint-fmt         - Format code and lint with auto-fixes"
	@echo "  make lint-check       - Check formatting and lint without modifying files"
	@echo "  make lint-typos       - Check for typos"
	@echo "  make lint-all         - Run formatting, linting, and type checking"
	@echo "  make security-bandit  - Run Bandit security scan"
	@echo "  make security-audit   - Run pip-audit dependency vulnerability scan"
	@echo "  make security         - Run all security scans"
	@echo "  make clean            - Clean build artifacts and cache"

sync:
	uv sync --all-groups --all-extras

test:
	uv run pytest tests -m "not integration"

test-integration:
	uv run pytest tests -m "integration" -v -o "addopts="

test-cov:
	uv run pytest tests -m "not integration" --cov=am_diag --cov-report=term-missing

test-ci:
	uv run pytest tests -m "not integration" \
		--cov=am_diag \
		--cov-report=xml \
		--cov-report=term-missing \
		--junitxml=pytest-results.xml -v

cov-report:
	uv run coverage xml
	uv run coverage html

cov: test-cov cov-report

lint-typing:
	uv run ty check am_diag/

lint-style:
	uv run ruff check .

lint-fmt:
	uv run ruff format .
	uv run ruff check --fix --unsafe-fixes .

lint-check:
	uv run ruff format --check .
	uv run ruff check .

lint-typos:
	uv run typos

lint-all: lint-fmt lint-typing lint-typos

security-bandit:
	uv run bandit -c pyproject.toml -r am_diag/ --severity-level high --confidence-level high

security-audit:
	uv run pip-audit --desc --ignore-vuln CVE-2025-69872 --ignore-vuln CVE-2026-3219 --ignore-vuln CVE-2026-6357

security: security-bandit security-audit

baml-gen:
	@echo "Regenerating BAML clients..."
	cd am_diag/llm && baml-cli generate --from baml_src/

clean:
	rm -rf .coverage coverage.xml htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -type d -name .ty_cache -exec rm -rf {} +
