"""
Chat service sub-package for the Prime Lands Real Estate Intelligence Platform.

Exposes the three intelligence-layer services and their supporting components
consumed by the application layer and Jupyter notebooks.

Modules:
    rag_service:  ``RAGService`` — LCEL-based retrieval-augmented generation
        chain with inline citation support; ``build_rag_chain`` factory.
    cag_cache:    ``CAGCache`` — two-tier semantic cache (FAQ pre-warming +
        conversation history) with cosine-similarity lookup (threshold 0.90).
    cag_service:  ``CAGService`` — cache-augmented generation that checks
        ``CAGCache`` before invoking the retriever, targeting < 500 ms latency.
    crag_service: ``CRAGService`` — corrective RAG that triggers re-retrieval
        when confidence scores fall below the configured threshold (0.6).
"""

from .rag_service import RAGService, build_rag_chain
from .cag_cache import CAGCache
from .cag_service import CAGService
from .crag_service import CRAGService

__all__ = [
    # RAG service
    "RAGService",
    "build_rag_chain",
    # CAG service
    "CAGCache",
    "CAGService",
    # CRAG service
    "CRAGService",
]