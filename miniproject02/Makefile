# =============================================================================
# Makefile – Prime Lands Intelligence Platform
# Placement: Makefile  (project root)
#
# Cross-platform (Linux / macOS / Windows via Git Bash or WSL)
# Usage:  make <target>
# =============================================================================

.PHONY: help setup crawl ingest test test-all test-integration test-cov \
        up up-dev down logs status full \
        lint format \
        clean clean-utils clean-submission

# ---------------------------------------------------------------------------
# Detect OS for cross-platform clean commands
# ---------------------------------------------------------------------------
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    RM_RF        = cmd /C "IF EXIST $(1) RMDIR /S /Q $(1)"
    RM_F         = cmd /C "IF EXIST $(1) DEL /F /Q $(1)"
    FIND_PYC     = cmd /C "FOR /R . %%X IN (*.pyc) DO DEL /F /Q \"%%X\""
    FIND_PYCACHE = cmd /C "FOR /D /R . %%X IN (__pycache__) DO RMDIR /S /Q \"%%X\""
else
    DETECTED_OS := $(shell uname -s)
    RM_RF        = rm -rf $(1)
    RM_F         = rm -f $(1)
    FIND_PYC     = find . -name "*.pyc" -delete
    FIND_PYCACHE = find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
endif

# ---------------------------------------------------------------------------
# Default target
# ---------------------------------------------------------------------------
help:
	@echo ""
	@echo "  Prime Lands Intelligence Platform"
	@echo "  ─────────────────────────────────"
	@echo ""
	@echo "  Development"
	@echo "    make setup           Install all dependencies"
	@echo "    make lint            Run ruff linter"
	@echo "    make format          Auto-format with ruff"
	@echo ""
	@echo "  Pipeline"
	@echo "    make crawl           Web crawler (notebook 01)"
	@echo "    make ingest          Chunking + indexing (notebook 02)"
	@echo "    make full            Full pipeline end-to-end"
	@echo ""
	@echo "  Testing"
	@echo "    make test            Unit tests only"
	@echo "    make test-all        Unit + integration tests"
	@echo "    make test-integration  Integration tests only"
	@echo "    make test-cov        Tests with HTML coverage report"
	@echo ""
	@echo "  Docker"
	@echo "    make up              Start all services (production)"
	@echo "    make up-dev          Start all services (dev profile)"
	@echo "    make down            Stop all services"
	@echo "    make logs            Tail API logs"
	@echo "    make status          Show service health"
	@echo ""
	@echo "  Maintenance"
	@echo "    make clean           Remove generated artefacts"
	@echo "    make clean-utils     Remove linter / type-checker caches"
	@echo "    make clean-submission  Full clean for final submission"
	@echo ""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup:
	bash deploy.sh setup

# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
crawl:
	bash deploy.sh crawl

ingest:
	bash deploy.sh ingest

## Run the full pipeline: crawl → ingest → test → launch services
full: crawl ingest test-all up
	@echo ""
	@echo "  ✅ Full pipeline complete."
	@echo ""

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

## Unit tests only (excludes integration markers)
test:
	pytest tests/ -v --tb=short -m "not integration"

## Unit + integration tests
test-all:
	pytest tests/ -v --tb=short

## Integration tests only
test-integration:
	pytest tests/ -v --tb=short -m integration

## Tests with HTML coverage report → data/coverage/index.html
test-cov:
	pytest tests/ -v --tb=short \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:data/coverage
	@echo ""
	@echo "  📊 Coverage report: data/coverage/index.html"
	@echo ""

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------
lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

# ---------------------------------------------------------------------------
# Docker operations
# ---------------------------------------------------------------------------

## Start all services (production mode, rebuilds images)
up:
	docker compose up -d --build

## Start all services (dev profile — exposes additional ports / volumes)
up-dev:
	docker compose --profile dev up -d --build

## Stop and remove all containers
down:
	docker compose down

## Tail live logs from the API service
logs:
	docker compose logs -f api

## Show health status of all running services
status:
	bash deploy.sh status

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

## Remove Python bytecode and pytest / coverage artefacts
clean:
	@echo "Cleaning Python cache and test artefacts..."
	$(FIND_PYCACHE)
	$(FIND_PYC)
	$(call RM_RF,.pytest_cache)
	$(call RM_RF,data/coverage)
	@echo "Done."

## Remove linter and type-checker caches
clean-utils:
	@echo "Cleaning linter and utility caches..."
	$(call RM_RF,.ruff_cache)
	$(call RM_RF,.mypy_cache)
	$(call RM_RF,.hypothesis)
	@echo "Done."

## Full clean — removes all generated data, caches, and evaluation reports.
## Use before packaging the project for submission.
clean-submission: clean clean-utils
	@echo "Workspace cleaned for final submission! (Data folder safely preserved)"
	$(call RM_RF,data/coverage)