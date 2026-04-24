# =============================================================================
# Nawaloka Hospital NL2SQL Platform — Cross-Platform Makefile
# Works on Windows (cmd/PowerShell), macOS, and Linux
# =============================================================================

# --- Detect OS ---
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    PYTHON       := python
    UV           := uv
    RM           := del /q
    RMDIR        := rmdir /s /q
    SEP          := \\
    NULL         := nul
    STREAMLIT_RUN := uv run streamlit run src\dashboard\app.py
else
    DETECTED_OS := $(shell uname -s)
    PYTHON       := python3
    UV           := uv
    RM           := rm -f
    RMDIR        := rm -rf
    SEP          := /
    NULL         := /dev/null
    STREAMLIT_RUN := uv run streamlit run src/dashboard/app.py
endif

.DEFAULT_GOAL := help

# =============================================================================
# HELP
# =============================================================================
.PHONY: help
help:
	@echo.
	@echo  ===============================================================
	@echo   Nawaloka NL2SQL Platform — Available Commands
	@echo   Detected OS: $(DETECTED_OS)
	@echo  ===============================================================
	@echo.
	@echo   Setup
	@echo     make install       Install all dependencies via uv
	@echo     make env           Copy .env example to .env (edit before use)
	@echo     make setup         Full first-time setup (install + env)
	@echo.
	@echo   Database
	@echo     make db-up         Start local PostgreSQL via Docker
	@echo     make db-down       Stop local PostgreSQL
	@echo     make db-seed       Seed the MediCore database
	@echo     make db-reset      Stop DB, remove volume, restart and reseed
	@echo.
	@echo   Run
	@echo     make run           Launch the Streamlit dashboard
	@echo.
	@echo   Test
	@echo     make test          Run all unit tests
	@echo     make test-verbose  Run tests with verbose output
	@echo.
	@echo   Observability
	@echo     make traces        Download LangFuse traces to traces/
	@echo.
	@echo   Clean
	@echo     make clean         Remove cache and temp files
	@echo     make clean-all     Clean + remove .venv
	@echo.

# =============================================================================
# SETUP
# =============================================================================
.PHONY: install
install:
	@echo [setup] Installing dependencies with uv...
	$(UV) sync
	@echo [setup] Done.

.PHONY: env
env:
ifeq ($(DETECTED_OS),Windows)
	@if not exist .env (copy ".env example" .env && echo [env] .env created. Edit it before running.) else (echo [env] .env already exists. Skipping.)
else
	@if [ ! -f .env ]; then cp ".env example" .env && echo "[env] .env created. Edit it before running."; else echo "[env] .env already exists. Skipping."; fi
endif

.PHONY: setup
setup: install env
	@echo [setup] First-time setup complete. Edit .env then run: make db-up db-seed run

# =============================================================================
# DATABASE
# =============================================================================
.PHONY: db-up
db-up:
	@echo [db] Starting PostgreSQL container...
	docker compose up -d
	@echo [db] PostgreSQL is running on port 5432.

.PHONY: db-down
db-down:
	@echo [db] Stopping PostgreSQL container...
	docker compose down
	@echo [db] Stopped.

.PHONY: db-seed
db-seed:
	@echo [db] Seeding MediCore database...
	$(UV) run python scripts/seed_supabase.py
	@echo [db] Seeding complete.

.PHONY: db-reset
db-reset:
	@echo [db] Resetting database (down + volume wipe + up + seed)...
	docker compose down -v
	docker compose up -d
	@echo [db] Waiting for PostgreSQL to be ready...
ifeq ($(DETECTED_OS),Windows)
	timeout /t 5 /nobreak > $(NULL)
else
	sleep 5
endif
	$(UV) run python scripts/seed_supabase.py
	@echo [db] Database reset complete.

# =============================================================================
# RUN
# =============================================================================
.PHONY: run
run:
	@echo [run] Starting Streamlit dashboard...
	@echo [run] Open http://localhost:8501 in your browser.
	$(STREAMLIT_RUN)

# =============================================================================
# TEST
# =============================================================================
.PHONY: test
test:
	@echo [test] Running unit tests...
	$(UV) run python -m pytest tests/
	@echo [test] Done.

.PHONY: test-verbose
test-verbose:
	@echo [test] Running unit tests (verbose)...
	$(UV) run python -m pytest tests/ -v

# =============================================================================
# OBSERVABILITY
# =============================================================================
.PHONY: traces
traces:
	@echo [traces] Downloading LangFuse traces...
	$(UV) run python scripts/download_traces.py
	@echo [traces] Traces saved to traces/.

# =============================================================================
# CLEAN
# =============================================================================
.PHONY: clean
clean:
	@echo [clean] Removing cache files...
ifeq ($(DETECTED_OS),Windows)
	@if exist __pycache__ $(RMDIR) __pycache__
	@if exist src\__pycache__ $(RMDIR) src\__pycache__
	@if exist .pytest_cache $(RMDIR) .pytest_cache
	@if exist langfuse_cache $(RMDIR) langfuse_cache
	@for /d /r . %%d in (__pycache__) do @if exist "%%d" $(RMDIR) "%%d" 2>$(NULL)
else
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>$(NULL) || true
	find . -type f -name "*.pyc" -delete 2>$(NULL) || true
	$(RMDIR) .pytest_cache langfuse_cache 2>$(NULL) || true
endif
	@echo [clean] Done.

.PHONY: clean-all
clean-all: clean
	@echo [clean] Removing virtual environment...
ifeq ($(DETECTED_OS),Windows)
	@if exist .venv $(RMDIR) .venv
else
	$(RMDIR) .venv 2>$(NULL) || true
endif
	@echo [clean-all] Done. Run make install to reinstall.