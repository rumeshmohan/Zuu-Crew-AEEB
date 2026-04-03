"""
Context Engineering - RAG System for Production

A modular, production-ready RAG system with multiple chunking strategies,
vector store management, and advanced retrieval techniques.
"""

__version__ = "1.0.0"
__author__ = "Context Engineering Team"

from .config import (
    DATA_DIR,
    CRAWL_OUT_DIR,
    VECTOR_DIR,
    MARKDOWN_DIR,
    CACHE_DIR,
    LOGS_DIR,
    CORPUS_FILE,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_SHOW_PROGRESS,
    CHAT_PROVIDER,
    CHAT_TIER,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_STREAMING,
    FIXED_CHUNK_SIZE,
    FIXED_CHUNK_OVERLAP,
    SEMANTIC_MAX_CHUNK_SIZE,
    SEMANTIC_MIN_CHUNK_SIZE,
    SLIDING_WINDOW_SIZE,
    SLIDING_STRIDE_SIZE,
    PARENT_CHUNK_SIZE,
    CHILD_CHUNK_SIZE,
    CHILD_OVERLAP,
    LATE_CHUNK_BASE_SIZE,
    LATE_CHUNK_SPLIT_SIZE,
    LATE_CHUNK_CONTEXT_WINDOW,
    TOP_K_RESULTS,
    SIMILARITY_THRESHOLD,
    CAG_CACHE_MAX_SIZE,
    CAG_SIMILARITY_THRESHOLD,
    CAG_HISTORY_TTL_HOURS,
    CRAG_CONFIDENCE_THRESHOLD,
    CRAG_EXPANDED_K,
    CRAWL_BASE_URL,
    CRAWL_MAX_DEPTH,
    CRAWL_MAX_PAGES,
    CRAWL_DELAY_SECONDS,
    validate,
    dump,
    get_config,
)

from .domain import (
    Document,
    Chunk,
    count_tokens,
    truncate_text,
)

from .infrastructure import (
    get_chat_llm,
    get_default_embeddings,
)

__all__ = [
    "__version__",
    "__author__",
    # Paths
    "DATA_DIR",
    "CRAWL_OUT_DIR",
    "VECTOR_DIR",
    "MARKDOWN_DIR",
    "CACHE_DIR",
    "LOGS_DIR",
    "CORPUS_FILE",
    # Embedding
    "EMBEDDING_PROVIDER",
    "EMBEDDING_MODEL",
    "EMBEDDING_BATCH_SIZE",
    "EMBEDDING_SHOW_PROGRESS",
    # Chat / LLM
    "CHAT_PROVIDER",
    "CHAT_TIER",
    "LLM_TEMPERATURE",
    "LLM_MAX_TOKENS",
    "LLM_STREAMING",
    # Chunking
    "FIXED_CHUNK_SIZE",
    "FIXED_CHUNK_OVERLAP",
    "SEMANTIC_MAX_CHUNK_SIZE",
    "SEMANTIC_MIN_CHUNK_SIZE",
    "SLIDING_WINDOW_SIZE",
    "SLIDING_STRIDE_SIZE",
    "PARENT_CHUNK_SIZE",
    "CHILD_CHUNK_SIZE",
    "CHILD_OVERLAP",
    "LATE_CHUNK_BASE_SIZE",
    "LATE_CHUNK_SPLIT_SIZE",
    "LATE_CHUNK_CONTEXT_WINDOW",
    # Retrieval
    "TOP_K_RESULTS",
    "SIMILARITY_THRESHOLD",
    # CAG
    "CAG_CACHE_MAX_SIZE",
    "CAG_SIMILARITY_THRESHOLD",
    "CAG_HISTORY_TTL_HOURS",
    # CRAG
    "CRAG_CONFIDENCE_THRESHOLD",
    "CRAG_EXPANDED_K",
    # Crawling
    "CRAWL_BASE_URL",
    "CRAWL_MAX_DEPTH",
    "CRAWL_MAX_PAGES",
    "CRAWL_DELAY_SECONDS",
    # Domain models
    "Document",
    "Chunk",
    # Utils
    "count_tokens",
    "truncate_text",
    # Infrastructure
    "get_chat_llm",
    "get_default_embeddings",
    # Config helpers
    "validate",
    "dump",
    "get_config",
]