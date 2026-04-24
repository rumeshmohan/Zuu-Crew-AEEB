# ================================================================
# Kapruka Gift-Concierge Agent – Makefile
# Cross-platform: Windows (PowerShell) & Linux/macOS
# ================================================================

# --- OS Detection ---------------------------------------------------
ifeq ($(OS),Windows_NT)
    PLATFORM   := windows
    CP         := powershell -Command "Copy-Item -Path"
    CP_END     := -Destination
    RM_CACHE   := powershell -Command "\
        Get-ChildItem -Recurse -Filter '__pycache__' -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue; \
        Get-ChildItem -Recurse -Filter '.ipynb_checkpoints' -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue; \
        Get-ChildItem -Recurse -Filter '*.pyc' | Remove-Item -Force -ErrorAction SilentlyContinue; \
        if (Test-Path 'qdrant_db') { Remove-Item -Recurse -Force 'qdrant_db' }"
    ZIP_CMD    := powershell -Command "Compress-Archive -Path '.\*' -DestinationPath '..\Kapruka_Gift_Concierge_Submission.zip' -Force"
    ZIP_MSG    := Submission zip created at ..\Kapruka_Gift_Concierge_Submission.zip
else
    PLATFORM   := linux
    RM_CACHE   := find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
                  find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null; \
                  find . -name "*.pyc" -delete 2>/dev/null; \
                  rm -rf qdrant_db/
    ZIP_CMD    := cd .. && zip -r Kapruka_Gift_Concierge_Submission.zip $(notdir $(CURDIR)) \
                      --exclude "*/.env" \
                      --exclude "*/.git/*" \
                      --exclude "*/data/catalog_pre_patch_backup.json"
    ZIP_MSG    := Submission zip created at ../Kapruka_Gift_Concierge_Submission.zip
endif
# --------------------------------------------------------------------

.PHONY: help install env crawl enrich patch-images ingest pipeline ui test test-v clean zip

# Default target
help:
	@echo ""
	@echo "  Kapruka Gift-Concierge Agent  [$(PLATFORM)]"
	@echo "  ============================================"
	@echo ""
	@echo "  Setup"
	@echo "    make install        Install dependencies via uv"
	@echo "    make env            Copy .env .example -> .env"
	@echo ""
	@echo "  Data Pipeline"
	@echo "    make crawl          Run the Playwright scraper (kapruka.com)"
	@echo "    make enrich         Run LLM enrichment / clean-and-patch"
	@echo "    make patch-images   Patch image_url into catalog.json (one-time)"
	@echo "    make ingest         Embed catalog into Qdrant (mode set in config/params.yaml)"
	@echo "    make pipeline       crawl + enrich + ingest (full first-time setup)"
	@echo ""
	@echo "  Run"
	@echo "    make ui             Launch the Streamlit Concierge UI"
	@echo ""
	@echo "  Testing"
	@echo "    make test           Run all tests via pytest"
	@echo "    make test-v         Run tests with verbose output"
	@echo ""
	@echo "  Maintenance"
	@echo "    make clean          Remove __pycache__, .ipynb_checkpoints, qdrant_db"
	@echo "    make zip            Clean + package project for submission"
	@echo ""

# ----------------------------------------------------------------
# Setup
# ----------------------------------------------------------------

install:
	uv sync

env:
ifeq ($(PLATFORM),windows)
	@powershell -Command "Copy-Item -Path '.env .example' -Destination '.env'"
else
	cp ".env .example" .env
endif
	@echo .env created – fill in your API keys before running.

# ----------------------------------------------------------------
# Data Pipeline
# ----------------------------------------------------------------

crawl:
	uv run python scraper/kapruka_crawler.py

enrich:
	uv run python scraper/clean_and_patch.py

ingest:
	uv run python memory/vector_db.py

pipeline: crawl enrich ingest
	@echo Full data pipeline complete : catalog scraped, enriched, and ingested.

# ----------------------------------------------------------------
# Run
# ----------------------------------------------------------------

ui:
	uv run streamlit run ui/app.py

# ----------------------------------------------------------------
# Testing
# ----------------------------------------------------------------

test:
	uv run pytest tests/

test-v:
	uv run pytest tests/ -v

# ----------------------------------------------------------------
# Maintenance
# ----------------------------------------------------------------

clean:
	@$(RM_CACHE)
	@echo Cleaned build artifacts and runtime cache.

zip: clean
	@$(ZIP_CMD)
	@echo $(ZIP_MSG)