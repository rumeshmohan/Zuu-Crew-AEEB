#!/usr/bin/env bash
# =============================================================================
# deploy.sh – Prime Lands Intelligence Platform deployment script
# =============================================================================

set -euo pipefail

# Colour helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()    { echo -e "${BLUE}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()     { error "$*"; exit 1; }

check_prereqs() {
  command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1 || die "Python is required"
  command -v docker >/dev/null 2>&1 || warn "Docker not found (only needed for 'up')"
}

# Cross-platform environment activator
activate_venv() {
  if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
  elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
  else
    die "Virtual environment not found! Run './deploy.sh setup' first."
  fi
}

cmd_setup() {
  info "Setting up Python environment..."
  pip install uv --quiet
  uv venv .venv
  activate_venv
  info "Installing dependencies..."
  uv pip install -r requirements.txt || uv pip install langchain langchain-openai qdrant-client pytest playwright # fallback
  info "Installing Playwright browsers..."
  playwright install chromium --with-deps
  success "Setup complete."
}

cmd_crawl() {
  info "Running property crawler..."
  activate_venv
  python -m src.crawlers.web_crawler
  success "Crawling complete."
}

cmd_ingest() {
  info "Running chunking and indexing pipeline..."
  activate_venv
  python -m src.ingestion.pipeline
  success "Ingestion complete."
}

cmd_test() {
  info "Running unit tests..."
  activate_venv
  pytest tests/ -v --tb=short -m "not integration"
  success "Unit tests passed."
}

cmd_test_all() {
  info "Running all tests (Unit + Integration)..."
  activate_venv
  pytest tests/ -v --tb=short
  success "All tests passed."
}

cmd_up() {
  info "Starting Docker services..."
  docker compose up -d --build
  success "Services started."
}

cmd_down() {
  info "Stopping Docker services..."
  docker compose down
  success "Services stopped."
}

cmd_logs() {
  docker compose logs -f api
}

cmd_status() {
  docker compose ps
}

cmd_full() {
  info "Running full pipeline..."
  cmd_setup
  cmd_crawl
  cmd_ingest
  cmd_test
  cmd_up
  success "Full pipeline complete!"
}

# Entrypoint
check_prereqs

COMMAND="${1:-help}"

case "$COMMAND" in
  setup)    cmd_setup ;;
  crawl)    cmd_crawl ;;
  ingest)   cmd_ingest ;;
  test)     cmd_test ;;
  test-all) cmd_test_all ;;
  up)       cmd_up ;;
  down)     cmd_down ;;
  logs)     cmd_logs ;;
  status)   cmd_status ;;
  full)     cmd_full ;;
  help|*)
    echo ""
    echo "  Prime Lands Intelligence Platform"
    echo ""
    echo "  Usage: ./deploy.sh [command]"
    echo ""
    echo "  Commands:"
    echo "    setup       Install Python deps + Playwright"
    echo "    crawl       Run web crawler"
    echo "    ingest      Run chunking + indexing"
    echo "    test        Run unit tests"
    echo "    test-all    Run unit + integration tests"
    echo "    up          Start Docker services"
    echo "    down        Stop Docker services"
    echo "    full        Full pipeline: setup → crawl → ingest → test → up"
    echo ""
    ;;
esac