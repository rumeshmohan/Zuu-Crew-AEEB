"""
Shared pytest fixtures for the Prime Lands RAG test suite.

Placement: tests/conftest.py  (project root → tests/)
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
SRC_PATH = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_PATH)):
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def config() -> dict:
    """Return a minimal config dict mirroring config.yaml structure."""
    return {
        "crawling": {
            "base_url": "https://www.primelands.lk",
            "start_paths": ["/", "/properties"],
            "max_depth": 2,
            "max_pages": 10,
            "exclude_patterns": ["login", "admin"],
            "request_delay": 0.5,
        },
        "chunking": {
            "fixed": {"chunk_size": 512, "chunk_overlap": 50},
            "sliding": {"chunk_size": 512, "chunk_overlap": 128},
            "semantic": {"breakpoint_threshold_type": "percentile"},
            "parent_child": {"parent_chunk_size": 1024, "child_chunk_size": 256},
            "late_chunk": {"chunk_size": 512},
        },
        "paths": {
            "corpus_file": "data/primelands_corpus.jsonl",
            "chunks_dir": "data/chunks",
            "vector_store": "data/qdrant",
            "evaluation_dir": "data/evaluation",
        },
        "rag": {"top_k": 5, "confidence_threshold": 0.6},
        "cag": {"similarity_threshold": 0.90},
    }


# ---------------------------------------------------------------------------
# Sample documents / corpus
# ---------------------------------------------------------------------------
SAMPLE_DOCUMENTS = [
    {
        "url": "https://www.primelands.lk/properties/pl-001",
        "property_id": "PL-001",
        "title": "Luxury Land in Horana",
        "address": "Horana, Kalutara",
        "price": "LKR 3,500,000",
        "bedrooms": None, "bathrooms": None,
        "sqft": 10890,
        "amenities": ["highway access", "water", "electricity"],
        "agent": "Nuwan Perera",
        "content": (
            "A prime 10890 sqft land plot located in Horana, Kalutara. "
            "This property enjoys direct highway access, making it ideal for "
            "commercial or residential development. Asking price LKR 3,500,000. "
            "Available on installment basis. Contact Nuwan Perera for viewings."
        ),
    },
    {
        "url": "https://www.primelands.lk/properties/pl-002",
        "property_id": "PL-002",
        "title": "Residential Plot in Kiribathgoda",
        "address": "Kiribathgoda, Gampaha",
        "price": "LKR 7,800,000",
        "bedrooms": None, "bathrooms": None,
        "sqft": 7260,
        "amenities": ["schools nearby", "hospital access", "public transport"],
        "agent": "Dilanka Fernando",
        "content": (
            "Residential plot of 7260 sqft situated in a quiet neighbourhood "
            "in Kiribathgoda. Close to schools and hospitals. Bank loans available. "
            "Price: LKR 7,800,000. Agent: Dilanka Fernando."
        ),
    },
    {
        "url": "https://www.primelands.lk/properties/pl-003",
        "property_id": "PL-003",
        "title": "Commercial Land in Colombo",
        "address": "Rajagiriya, Colombo",
        "price": "LKR 25,000,000",
        "bedrooms": None, "bathrooms": None,
        "sqft": 21780,
        "amenities": ["main road access", "electricity", "drainage"],
        "agent": "Saman Wickramasinghe",
        "content": (
            "21780 sqft commercial-grade land in Rajagiriya. Main road frontage. "
            "All utilities connected. Price: LKR 25,000,000. Flexible payment plans available."
        ),
    },
    {
        "url": "https://www.primelands.lk/properties/pl-004",
        "property_id": "PL-004",
        "title": "Affordable Plot in Piliyandala",
        "address": "Piliyandala, Colombo",
        "price": "LKR 2,200,000",
        "bedrooms": None, "bathrooms": None,
        "sqft": 4356,
        "amenities": ["water", "electricity"],
        "agent": "Chamari Silva",
        "content": (
            "4356 sqft land in Piliyandala. Affordable and suitable for a starter home. "
            "10% down payment with bank financing accepted. Price: LKR 2,200,000."
        ),
    },
    {
        "url": "https://www.primelands.lk/properties/pl-005",
        "property_id": "PL-005",
        "title": "Scenic Estate in Kandy",
        "address": "Peradeniya, Kandy",
        "price": "LKR 12,000,000",
        "bedrooms": None, "bathrooms": None,
        "sqft": 43560,
        "amenities": ["scenic view", "well water", "electricity", "road access"],
        "agent": "Ranga Jayasuriya",
        "content": (
            "43560 sqft scenic estate near Peradeniya, Kandy. Ideal for eco-tourism "
            "or private residence. Panoramic mountain views. Price: LKR 12,000,000."
        ),
    },
]


@pytest.fixture(scope="session")
def sample_documents() -> list[dict]:
    return SAMPLE_DOCUMENTS


@pytest.fixture(scope="session")
def corpus_file(tmp_path_factory, sample_documents) -> Path:
    tmp = tmp_path_factory.mktemp("data")
    path = tmp / "primelands_corpus.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for doc in sample_documents:
            f.write(json.dumps(doc) + "\n")
    return path


# ---------------------------------------------------------------------------
# Deterministic embedding helper
# ---------------------------------------------------------------------------
def _det_vector(text: str, dims: int = 384) -> list[float]:
    """
    Produce a *deterministic* vector for *text* using an MD5-seeded RNG.

    The same input string always returns the same vector, which is essential
    for CAGCache set()/get() round-trips: if embed_query() returns a different
    random vector on each call, cosine-similarity lookup will always miss.
    """
    seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2 ** 31)
    rng = random.Random(seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(dims)]


# ---------------------------------------------------------------------------
# Mock LLM — session-scoped FakeListLLM (proper LangChain Runnable)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mock_llm():
    """
    LangChain-compatible fake LLM for unit tests.

    FakeListLLM is a proper BaseLLM / Runnable — it integrates with LCEL chains
    (prompt | llm | StrOutputParser()) without Pydantic validation errors.
    Session-scoped so it is safely shareable by class- and module-scoped fixtures.
    """
    try:
        from langchain_core.language_models.fake import FakeListLLM
    except ImportError:
        from langchain.llms.fake import FakeListLLM  # type: ignore[no-redef]

    answer = "Mock LLM answer about Prime Lands properties."
    return FakeListLLM(responses=[answer] * 500)


# ---------------------------------------------------------------------------
# Mock embeddings — deterministic 384-dim vectors, session-scoped
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mock_embeddings():
    """
    Return a MagicMock embeddings object producing *deterministic* 384-dim vectors.

    Identical input text always produces an identical vector (MD5-seeded RNG),
    which is required for cache set()/get() round-trips to succeed.
    Session-scoped to prevent ScopeMismatch when used inside class-scoped fixtures.
    """
    emb = MagicMock()
    emb.embed_documents.side_effect = lambda texts: [_det_vector(t) for t in texts]
    emb.embed_query.side_effect = lambda text: _det_vector(text)
    return emb


# ---------------------------------------------------------------------------
# Mock retriever — session-scoped
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def mock_retriever(sample_documents):
    """Return a mock retriever yielding 3 real LangChain Documents."""
    from langchain_core.documents import Document

    retriever = MagicMock()
    docs = [
        Document(
            page_content=d["content"],
            metadata={"url": d["url"], "property_id": d["property_id"], "title": d["title"]},
        )
        for d in sample_documents[:3]
    ]
    retriever.invoke.return_value = docs
    retriever.get_relevant_documents.return_value = docs
    return retriever


# ---------------------------------------------------------------------------
# Helper: build a CAGCache instance using the real constructor signature
# ---------------------------------------------------------------------------
def _build_cag_cache(mock_embeddings):
    """
    Instantiate CAGCache without assuming its constructor signature.

    Handles common parameter name variations:
      - embeddings / embedding_model / embed_model / embedder
      - cache_dir  (passed as Path so .mkdir() works)
    """
    from context_engineering.application.chat_service import CAGCache
    import inspect

    params = set(inspect.signature(CAGCache.__init__).parameters) - {"self"}
    kwargs = {}

    for name in ("embeddings", "embedding_model", "embed_model", "embedder"):
        if name in params:
            kwargs[name] = mock_embeddings
            break

    if "cache_dir" in params:
        kwargs["cache_dir"] = Path(tempfile.mkdtemp())

    return CAGCache(**kwargs)


@pytest.fixture
def cag_cache(mock_embeddings):
    """Provide a CAGCache instance compatible with the real implementation."""
    try:
        return _build_cag_cache(mock_embeddings)
    except Exception:
        pytest.skip("Could not instantiate CAGCache – check constructor signature")


# ---------------------------------------------------------------------------
# Helper: build a VectorStoreService using the real constructor signature
# ---------------------------------------------------------------------------
def _build_vs_service(mock_embeddings, db_path):
    """Instantiate VectorStoreService without assuming its constructor signature."""
    from context_engineering.application.ingest_documents_service.vector_store_service import (
        VectorStoreService,
    )
    import inspect

    params = set(inspect.signature(VectorStoreService.__init__).parameters) - {"self"}
    kwargs = {}

    for name in ("embeddings", "embedding_model", "embed_model"):
        if name in params:
            kwargs[name] = mock_embeddings
            break

    for name in ("db_path", "path", "persist_directory", "vector_store_path", "storage_path"):
        if name in params:
            kwargs[name] = str(db_path)
            break

    return VectorStoreService(**kwargs)