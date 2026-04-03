"""
Ingest documents sub-package for the Prime Lands Real Estate Intelligence Platform.

Exposes the web crawling and text chunking components used in Part 1 (The
Property Crawler) and Part 2 (The Chunking Lab) to build and index the
Prime Lands corpus.

Modules:
    chunkers:     ``ChunkingService`` — orchestrates all five chunking
                  strategies; ``semantic_chunk``, ``fixed_chunk``,
                  ``sliding_chunk``, ``parent_child_chunk``, and ``late_chunk``
                  — individual strategy implementations with token-aware
                  splitting; ``late_chunk_split`` — post-embedding segment
                  helper for late chunking; ``count_tokens`` — tiktoken-based
                  token counter shared across strategies.
    web_crawler:  ``PrimeLandsWebCrawler`` — async Playwright BFS crawler that
                  extracts property metadata and saves the corpus as Markdown
                  files and a JSONL corpus file.
"""

from .chunkers import (
    ChunkingService,
    semantic_chunk,
    fixed_chunk,
    sliding_chunk,
    parent_child_chunk,
    late_chunk,
    late_chunk_split,
    count_tokens,
)
from .web_crawler import PrimeLandsWebCrawler

__all__ = [
    # Chunking service and strategies
    "ChunkingService",
    "semantic_chunk",
    "fixed_chunk",
    "sliding_chunk",
    "parent_child_chunk",
    "late_chunk",
    "late_chunk_split",
    # Token utility
    "count_tokens",
    # Web crawler
    "PrimeLandsWebCrawler",
]