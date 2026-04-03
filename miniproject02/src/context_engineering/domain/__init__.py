"""
Domain layer for the Prime Lands Real Estate Intelligence Platform.

This package exposes the canonical data models and shared utility functions
that form the core business logic consumed by the application service layer
(RAGService, CAGService, CRAGService, ChunkingService).

Modules:
    models:  Dataclass definitions for ``Document``, ``Chunk``, ``Evidence``,
             ``RAGQuery``, and ``RAGResponse``.
    utils:   Pipeline helpers — document formatting, confidence scoring,
             citation extraction, text truncation, and token counting.
    prompts: LangChain prompt templates for RAG, CAG, and CRAG chains.
    tools:   Custom LangChain tools used by the agent layer.
"""

from .models import Document, Chunk, Evidence, RAGQuery, RAGResponse
from .utils import (
    format_docs,
    calculate_confidence,
    extract_citations,
    truncate_text,
    count_tokens,
)

__all__ = [
    # Models
    "Document",
    "Chunk",
    "Evidence",
    "RAGQuery",
    "RAGResponse",
    # Utils
    "format_docs",
    "calculate_confidence",
    "extract_citations",
    "truncate_text",
    "count_tokens",
]