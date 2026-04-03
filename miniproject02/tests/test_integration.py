"""
End-to-end integration smoke test.

Chains: corpus → chunker → vector store → RAGService → answer

Placement: tests/test_integration.py

Run with:
    pytest tests/test_integration.py -v -m integration
"""

from __future__ import annotations

import inspect
import tempfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration  # tag so unit-only runs can skip these


@pytest.fixture(scope="module")
def all_services():
    """Import everything needed for the integration test."""
    try:
        from context_engineering.application.chat_service import (
            CAGCache, CAGService, CRAGService, RAGService,
        )
        from context_engineering.application.ingest_documents_service.chunkers import fixed_chunk
        from context_engineering.application.ingest_documents_service.vector_store_service import (
            VectorStoreService,
        )
        return {
            "fixed_chunk": fixed_chunk,
            "VectorStoreService": VectorStoreService,
            "RAGService": RAGService,
            "CAGService": CAGService,
            "CRAGService": CRAGService,
            "CAGCache": CAGCache,
        }
    except ImportError:
        pytest.skip("Services not importable for integration tests")


def _build_vs(VectorStoreService, mock_embeddings, db_path):
    """Instantiate VectorStoreService by introspecting its real constructor."""
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


def _build_cag_cache(CAGCache, mock_embeddings):
    """Instantiate CAGCache by introspecting its real constructor."""
    params = set(inspect.signature(CAGCache.__init__).parameters) - {"self"}
    kwargs = {}

    for name in ("embeddings", "embedding_model", "embed_model", "embedder"):
        if name in params:
            kwargs[name] = mock_embeddings
            break

    if "cache_dir" in params:
        kwargs["cache_dir"] = Path(tempfile.mkdtemp())

    return CAGCache(**kwargs)


def _build_cag_service(CAGService, retriever, llm, cache, k=3, rag_service=None):
    """Instantiate CAGService by introspecting its real constructor."""
    params = set(inspect.signature(CAGService.__init__).parameters) - {"self"}
    kwargs = {}

    for name in ("retriever", "vector_store", "vector_retriever", "vs"):
        if name in params:
            kwargs[name] = retriever
            break

    for name in ("llm", "language_model", "model", "chat_model"):
        if name in params:
            kwargs[name] = llm
            break

    for name in ("cache", "cag_cache", "response_cache"):
        if name in params:
            kwargs[name] = cache
            break

    for name in ("k", "top_k", "num_results"):
        if name in params:
            kwargs[name] = k
            break

    # Some CAGService implementations delegate to an underlying RAGService
    for name in ("rag_service", "rag", "base_service", "base_rag"):
        if name in params:
            kwargs[name] = rag_service
            break

    return CAGService(**kwargs)


class TestEndToEndPipeline:
    """Smoke test: build a mini RAG pipeline from scratch.

    The `pipeline` fixture is class-scoped so all tests share one built
    pipeline (avoids repeated indexing).  mock_embeddings and mock_llm are
    session-scoped in conftest.py, satisfying pytest's scope-nesting rule
    (session >= class).
    """

    @pytest.fixture(scope="class")
    def pipeline(self, all_services, sample_documents, mock_embeddings, mock_llm, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("integration")

        # 1. Chunk
        chunks = all_services["fixed_chunk"](sample_documents)
        assert len(chunks) > 0, "fixed_chunk produced no output"

        # 2. Index — introspect constructor so we are not brittle to param names
        vs = _build_vs(all_services["VectorStoreService"], mock_embeddings, tmp / "qdrant")

        if not hasattr(vs, "index_chunks"):
            pytest.skip(
                "VectorStoreService.index_chunks() not implemented – "
                "add the method before running integration tests"
            )
        vs.index_chunks(chunks, collection_name="primelands_integration")

        # 3. Retriever
        # Wrap mock_embeddings in a proper LangChain Embeddings subclass so that
        # QdrantVectorStore's isinstance() validation passes even for plain mocks.
        from langchain_qdrant import QdrantVectorStore
        from langchain_core.embeddings import Embeddings as LCEmbeddings

        class _WrappedEmbeddings(LCEmbeddings):
            def __init__(self, inner):
                self._inner = inner

            def embed_documents(self, texts):
                return self._inner.embed_documents(texts)

            def embed_query(self, text):
                return self._inner.embed_query(text)

        safe_embeddings = (
            mock_embeddings
            if isinstance(mock_embeddings, LCEmbeddings)
            else _WrappedEmbeddings(mock_embeddings)
        )

        qs = QdrantVectorStore(
            client=vs.client,
            collection_name="primelands_integration",
            embedding=safe_embeddings,
        )
        retriever = qs.as_retriever(search_kwargs={"k": 3})

        # 4. Services
        rag = all_services["RAGService"](retriever=retriever, llm=mock_llm, k=3)
        cache = _build_cag_cache(all_services["CAGCache"], mock_embeddings)
        # Pass rag as rag_service so CAGService implementations that wrap RAG work too
        cag = _build_cag_service(
            all_services["CAGService"], retriever, mock_llm, cache, k=3, rag_service=rag
        )
        crag = all_services["CRAGService"](retriever=retriever, llm=mock_llm, initial_k=3, expanded_k=5)

        return {"rag": rag, "cag": cag, "crag": crag}

    def test_rag_answers_query(self, pipeline):
        result = pipeline["rag"].generate("What properties are available in Horana?")
        assert isinstance(result, dict)

    def test_cag_second_query_is_cache_hit(self, pipeline):
        q = "What are the payment plans?"
        pipeline["cag"].generate(q)
        r2 = pipeline["cag"].generate(q)
        assert r2.get("cache_hit") or r2.get("from_cache"), (
            "Second identical query should be a cache hit"
        )

    def test_crag_returns_confidence(self, pipeline):
        result = pipeline["crag"].generate("Show me commercial land in Colombo")
        accepted_keys = (
            "confidence", "confidence_score", "score",
            "confidence_final", "confidence_initial",
        )
        has_conf = any(k in result for k in accepted_keys)
        assert has_conf, f"CRAG response missing confidence score. Keys: {list(result.keys())}"

    def test_full_pipeline_produces_non_empty_answers(self, pipeline):
        query = "Are there properties with bank loan facilities?"
        for name, svc in pipeline.items():
            result = svc.generate(query)
            answer = result.get("answer") or result.get("response") or result.get("output") or ""
            assert answer.strip(), f"{name}: empty answer for query '{query}'"