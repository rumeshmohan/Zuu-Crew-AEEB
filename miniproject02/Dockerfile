# =============================================================================
# Prime Lands Real Estate Intelligence Platform
# Placement: Dockerfile  (project root)
# =============================================================================

FROM python:3.11-slim AS base

LABEL maintainer="Prime Lands AI Team"
LABEL description="Real Estate Intelligence Platform – RAG API"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Builder stage – install Python packages
# ---------------------------------------------------------------------------
FROM base AS builder

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv --quiet

COPY pyproject.toml .
COPY uv.lock* ./

# Install all production deps into a virtual env
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python \
        langchain \
        langchain-community \
        langchain-text-splitters \
        langchain-qdrant \
        langchain-huggingface \
        langchain-openai \
        langchain-cohere \
        qdrant-client \
        playwright \
        beautifulsoup4 \
        markdownify \
        sentence-transformers \
        tiktoken \
        fastapi \
        uvicorn[standard] \
        pydantic \
        pydantic-settings \
        python-dotenv \
        pyyaml \
        pandas \
        numpy \
        scikit-learn \
        cohere \
        nest-asyncio

# Install Playwright browsers
RUN /app/.venv/bin/playwright install chromium --with-deps

# ---------------------------------------------------------------------------
# Runtime stage – minimal final image
# ---------------------------------------------------------------------------
FROM base AS runtime

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy source
COPY src/ ./src/
COPY config/ ./config/
COPY api/ ./api/

# Create data directories (volumes will override in production)
RUN mkdir -p /app/data/chunks /app/data/qdrant /app/data/evaluation /app/data/markdown

# Non-root user for security
RUN useradd -m -u 1001 primelands && chown -R primelands:primelands /app
USER primelands

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
