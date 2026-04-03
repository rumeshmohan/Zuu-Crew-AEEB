"""
Application layer for the Prime Lands Real Estate Intelligence Platform.

This is the top-level package that wires together the ingest and chat service
sub-packages, providing a single import surface for Jupyter notebooks and any
external consumers of the pipeline.

Modules:
    ingest_documents_service: ``PrimeLandsWebCrawler`` — async Playwright BFS
        crawler for corpus collection; ``ChunkingService`` — orchestrates the
        five token-aware chunking strategies and Qdrant indexing.
    chat_service: ``RAGService`` — LCEL retrieval-augmented generation with
        inline citations; ``CAGService`` / ``CAGCache`` — cache-augmented
        generation with two-tier semantic caching; ``CRAGService`` — corrective
        RAG with confidence-threshold-gated supplementary retrieval;
        ``build_rag_chain`` — factory function for the LCEL chain.
"""

from .ingest_documents_service import (
    ChunkingService,
    PrimeLandsWebCrawler,
)
from .chat_service import (
    RAGService,
    CAGService,
    CRAGService,
    CAGCache,
    build_rag_chain,
)

__all__ = [
    # Ingest services
    "ChunkingService",
    "PrimeLandsWebCrawler",
    # Chat services
    "RAGService",
    "CAGService",
    "CRAGService",
    "CAGCache",
    "build_rag_chain",
]