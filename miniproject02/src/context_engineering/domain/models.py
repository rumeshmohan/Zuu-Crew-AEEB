"""
Core domain models for the Prime Lands Real Estate Intelligence Platform.

Defines the canonical data structures shared across the crawling, chunking,
and RAG/CAG/CRAG service layers.  All models are plain ``dataclass`` objects
with lightweight validation in ``__post_init__`` where invariants must be
enforced at construction time.

Classes:
    Document:    A crawled web page with its full Markdown content.
    Chunk:       A single text segment produced by a chunking strategy.
    Evidence:    A retrieved excerpt surfaced during RAG retrieval.
    RAGQuery:    Encapsulates a user question and retrieval parameters.
    RAGResponse: The complete output of a RAG/CAG/CRAG inference call.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass
class Document:
    """A crawled web document from the Prime Lands corpus.

    Produced by ``PrimeLandsWebCrawler`` and consumed by ``ChunkingService``
    as the primary ingestion unit.

    Args:
        url (str): Canonical source URL of the page.  Must be non-empty.
        title (str): Page or listing title extracted from the DOM.
        content (str): Full page text rendered as Markdown.  Must be non-empty.
        metadata (Dict[str, Any]): Arbitrary crawler-populated fields such as
            crawl depth, outbound links, or extracted headings.

    Raises:
        ValueError: If ``url`` or ``content`` is an empty string.
    """

    url: str
    title: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.url:
            raise ValueError("Document URL cannot be empty")
        if not self.content:
            raise ValueError("Document content cannot be empty")


@dataclass
class Chunk:
    """A text segment produced by one of the five chunking strategies.

    Created by ``ChunkingService`` and stored as a Qdrant vector payload.
    The ``strategy`` field is validated against the supported set on
    construction.

    Args:
        text (str): The chunk's raw text content.
        strategy (str): Name of the chunking strategy that produced this chunk.
            Must be one of ``"semantic"``, ``"fixed"``, or ``"sliding"``.
        chunk_index (int): Zero-based position of this chunk within its source
            document.
        url (str): Source document URL, preserved for citation.
        title (str): Source document title, preserved for citation.
        metadata (Dict[str, Any]): Strategy-specific payload (e.g. parent chunk
            ID for parent-child, sentence boundaries for semantic).

    Raises:
        ValueError: If ``strategy`` is not a recognised chunking strategy name.
    """

    text: str
    strategy: str
    chunk_index: int
    url: str
    title: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.strategy not in ["semantic", "fixed", "sliding"]:
            raise ValueError(f"Invalid strategy: {self.strategy}")


@dataclass
class Evidence:
    """A retrieved text excerpt used to ground an LLM-generated answer.

    Populated by ``RAGService``, ``CAGService``, and ``CRAGService`` from
    Qdrant search results and attached to every ``RAGResponse`` for
    auditability and inline citation rendering.

    Args:
        url (str): Source page URL, used to render inline ``[url]`` citations.
        title (str): Human-readable source title.
        quote (str): Representative text excerpt (typically the first ~400
            characters of the matched chunk).
        strategy (str): Chunking strategy of the matched chunk.
        score (Optional[float]): Cosine similarity or relevance score returned
            by the vector store.  ``None`` when not available.
        metadata (Dict[str, Any]): Additional retrieval payload (e.g. chunk
            index, collection name).
    """

    url: str
    title: str
    quote: str
    strategy: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGQuery:
    """Query parameters for a single RAG/CAG/CRAG inference call.

    Acts as a typed configuration object passed into the service layer,
    keeping caller code free of scattered keyword arguments.

    Args:
        query (str): The user's natural-language question.
        k (int): Number of documents to retrieve from the vector store.
            Defaults to ``4``.
        confidence_threshold (float): Minimum confidence score required before
            ``CRAGService`` accepts retrieved context without correction.
            Defaults to ``0.6``.
        use_cache (bool): When ``True``, ``CAGService`` checks the two-tier
            FAQ and history cache before invoking the retriever.
            Defaults to ``True``.
    """

    query: str
    k: int = 4
    confidence_threshold: float = 0.6
    use_cache: bool = True


@dataclass
class RAGResponse:
    """The complete output of a RAG, CAG, or CRAG inference call.

    Returned by all three service classes so that callers and evaluation
    harnesses can treat responses uniformly regardless of which pipeline
    produced them.

    Args:
        answer (str): LLM-generated answer text, which may contain inline
            ``[url]`` citations.
        evidence (list[Evidence]): Ordered list of retrieved evidence objects
            used to ground the answer.
        confidence (Optional[float]): Retrieval confidence score computed by
            ``calculate_confidence``; present only for CRAG responses.
        cache_hit (bool): ``True`` when the answer was served from the CAG
            cache rather than generated live.  Defaults to ``False``.
        generation_time (float): Wall-clock seconds from query receipt to
            response completion.  Defaults to ``0.0``.
        metadata (Dict[str, Any]): Supplementary pipeline metadata, e.g.
            ``{"correction_applied": True}`` for CRAG or cache tier for CAG.
    """

    answer: str
    evidence: list[Evidence]
    confidence: Optional[float] = None
    cache_hit: bool = False
    generation_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)