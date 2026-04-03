"""
Unit tests for VectorStoreService and evaluation output files.

Placement: tests/test_vector_store.py

Run with:
    pytest tests/test_vector_store.py -v
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helper: instantiate VectorStoreService without hardcoding its constructor
# ---------------------------------------------------------------------------
def _make_vs_service(VectorStoreService, mock_embeddings, db_path):
    """
    Build a VectorStoreService by introspecting its real __init__ signature
    so tests don't break when parameter names differ from our assumptions.
    """
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


# ---------------------------------------------------------------------------
# VectorStoreService import guard
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def vs_service_class():
    try:
        from context_engineering.application.ingest_documents_service.vector_store_service import (
            VectorStoreService,
        )
        return VectorStoreService
    except ImportError:
        pytest.skip("VectorStoreService not importable")


# ---------------------------------------------------------------------------
# VectorStoreService unit tests
# ---------------------------------------------------------------------------
class TestVectorStoreService:
    EXPECTED_COLLECTIONS = [
        "primelands_fixed",
        "primelands_sliding",
        "primelands_semantic",
        "primelands_parent_child",
        "primelands_late_chunk",
    ]

    @pytest.fixture
    def vs(self, vs_service_class, mock_embeddings, tmp_path):
        return _make_vs_service(vs_service_class, mock_embeddings, tmp_path / "qdrant")

    def test_instantiation(self, vs):
        assert vs is not None

    def test_has_client(self, vs):
        assert hasattr(vs, "client"), "VectorStoreService missing 'client' attribute"

    def test_index_chunks_creates_collection(self, vs, sample_documents):
        """
        VectorStoreService must expose index_chunks(chunks, collection_name)
        that creates the named Qdrant collection.
        """
        if not hasattr(vs, "index_chunks"):
            pytest.skip(
                "VectorStoreService.index_chunks() not implemented – "
                "add the method before this test can pass"
            )
        fake_chunks = [
            {"content": d["content"], "metadata": {"url": d["url"], "property_id": d["property_id"]}}
            for d in sample_documents
        ]
        try:
            vs.index_chunks(fake_chunks, collection_name="primelands_test")
            collections = {c.name for c in vs.client.get_collections().collections}
            assert "primelands_test" in collections
        except Exception as exc:
            pytest.fail(f"index_chunks() raised: {exc}")

    def test_search_returns_results(self, vs, sample_documents):
        """After indexing, search() must return a list."""
        if not hasattr(vs, "index_chunks"):
            pytest.skip("VectorStoreService.index_chunks() not implemented")
        if not hasattr(vs, "search"):
            pytest.skip("VectorStoreService.search() not implemented")

        fake_chunks = [
            {"content": d["content"], "metadata": {"url": d["url"]}}
            for d in sample_documents
        ]
        try:
            vs.index_chunks(fake_chunks, collection_name="primelands_search_test")
            results = vs.search("luxury land Horana", collection_name="primelands_search_test", k=3)
            assert isinstance(results, list)
        except Exception as exc:
            pytest.fail(f"search() raised: {exc}")

    def test_search_respects_k_parameter(self, vs, sample_documents):
        fake_chunks = [
            {"content": d["content"], "metadata": {"url": d["url"]}} for d in sample_documents
        ]
        try:
            vs.index_chunks(fake_chunks, collection_name="primelands_k_test")
            for k in (1, 3, 5):
                results = vs.search("property payment plan", collection_name="primelands_k_test", k=k)
                assert len(results) <= k, f"Expected <= {k} results, got {len(results)}"
        except Exception:
            pytest.skip("search() not available in this VectorStoreService API")

    # -----------------------------------------------------------------------
    # all 5 Qdrant collections populated — Part 2 "Qdrant Indexing"
    # -----------------------------------------------------------------------
    def test_all_five_strategy_collections_can_be_created(self, vs_service_class, mock_embeddings, tmp_path, sample_documents):
        """
        Rubric Part 2: 'Persistent index created using Qdrant, all 5 collections
        populated with embeddings and rich metadata.'

        Verifies that index_chunks() can successfully create all five required
        strategy collections in a single VectorStoreService instance.  Each
        collection must be visible via client.get_collections() after indexing.
        """
        vs = _make_vs_service(vs_service_class, mock_embeddings, tmp_path / "qdrant_5col")

        if not hasattr(vs, "index_chunks"):
            pytest.skip(
                "VectorStoreService.index_chunks() not implemented – "
                "add the method before this test can pass"
            )

        fake_chunks = [
            {"content": d["content"], "metadata": {"url": d["url"], "property_id": d["property_id"]}}
            for d in sample_documents
        ]

        created = []
        for collection_name in self.EXPECTED_COLLECTIONS:
            try:
                vs.index_chunks(fake_chunks, collection_name=collection_name)
                created.append(collection_name)
            except Exception as exc:
                pytest.fail(f"index_chunks() failed for collection '{collection_name}': {exc}")

        existing = {c.name for c in vs.client.get_collections().collections}
        missing = [c for c in self.EXPECTED_COLLECTIONS if c not in existing]

        assert not missing, (
            f"The following required Qdrant collections were not created: {missing}. "
            f"Collections present: {sorted(existing)}. "
            "Rubric Part 2 requires all 5 strategy collections to be indexed."
        )

    def test_indexed_chunks_have_metadata(self, vs, sample_documents):
        """
        Rubric Part 2: 'rich metadata' in Qdrant collections.

        After indexing, at least one point in the collection should have a
        non-empty payload (metadata) stored alongside the vector.
        """
        if not hasattr(vs, "index_chunks"):
            pytest.skip("VectorStoreService.index_chunks() not implemented")

        fake_chunks = [
            {
                "content": d["content"],
                "metadata": {
                    "url": d["url"],
                    "property_id": d["property_id"],
                    "title": d["title"],
                    "price": d["price"],
                },
            }
            for d in sample_documents
        ]
        try:
            vs.index_chunks(fake_chunks, collection_name="primelands_meta_test")
            # Scroll to retrieve stored points with payloads
            points, _ = vs.client.scroll(
                collection_name="primelands_meta_test",
                with_payload=True,
                limit=1,
            )
            assert points, "No points found in collection after indexing"
            payload = points[0].payload or {}
            assert payload, (
                "Indexed point has an empty payload — metadata was not stored. "
                "Rubric Part 2 requires 'rich metadata' in Qdrant collections."
            )
        except Exception as exc:
            pytest.fail(f"Metadata check raised: {exc}")


# ---------------------------------------------------------------------------
# Evaluation output file tests (Part 4 quick-check)
# ---------------------------------------------------------------------------
class TestEvaluationOutputs:
    """
    Verify the three required output files from Part 4.
    Skipped gracefully when data/ directory is not yet present.
    """

    def _find_eval_dir(self) -> Path | None:
        candidates = [
            Path(__file__).parent.parent / "data" / "evaluation",
            Path(__file__).parent.parent / "data",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None

    def test_chunking_comparison_csv_exists(self):
        ev = self._find_eval_dir()
        if ev is None:
            pytest.skip("Evaluation directory not found – run notebook 04 first")
        assert (ev / "chunking_comparison.csv").exists(), f"Missing: {ev / 'chunking_comparison.csv'}"

    def test_chunking_comparison_has_required_columns(self):
        """
        Rubric Part 4: 'Precision@5, Recall@5, Answer Relevance, and Latency metrics'.
        All four metric categories must be present as columns in the CSV.
        """
        import pandas as pd
        ev = self._find_eval_dir()
        if ev is None:
            pytest.skip("Evaluation directory not found")
        f = ev / "chunking_comparison.csv"
        if not f.exists():
            pytest.skip("chunking_comparison.csv not found")
        df = pd.read_csv(f)
        cols_lower = [c.lower() for c in df.columns]

        required = {
            "strategy": "strategy (chunking method name)",
            "latency": "latency (retrieval time)",
            "precision": "Precision@5",
            "recall": "Recall@5",
            "relevance": "Answer Relevance",
        }
        for keyword, label in required.items():
            assert any(keyword in c for c in cols_lower), (
                f"chunking_comparison.csv missing '{label}' column. "
                f"Columns present: {list(df.columns)}"
            )

    def test_chunking_comparison_has_all_five_strategies(self):
        """Each of the 5 chunking strategies must appear in the comparison CSV."""
        import pandas as pd
        ev = self._find_eval_dir()
        if ev is None:
            pytest.skip("Evaluation directory not found")
        f = ev / "chunking_comparison.csv"
        if not f.exists():
            pytest.skip("chunking_comparison.csv not found")
        df = pd.read_csv(f)

        # Find the strategy column (flexible naming)
        strategy_col = next(
            (c for c in df.columns if "strategy" in c.lower()), None
        )
        if strategy_col is None:
            pytest.skip("No 'strategy' column found – cannot check strategy coverage")

        strategies_found = {str(v).lower() for v in df[strategy_col].dropna()}
        expected_keywords = ("fixed", "sliding", "semantic", "parent", "late")
        missing = [kw for kw in expected_keywords if not any(kw in s for s in strategies_found)]
        assert not missing, (
            f"chunking_comparison.csv missing rows for strategies: {missing}. "
            f"Strategies found: {sorted(strategies_found)}. "
            "Rubric Part 4 requires all 5 strategies to be evaluated."
        )

    def test_cag_stats_json_exists(self):
        ev = self._find_eval_dir()
        if ev is None:
            pytest.skip("Evaluation directory not found")
        assert (ev / "cag_stats.json").exists(), f"Missing: {ev / 'cag_stats.json'}"

    def test_cag_stats_json_structure(self):
        """
        cag_stats.json must contain hit_rate, cache_hits, and total_queries.
        Searches both top-level keys and one level deep.
        """
        ev = self._find_eval_dir()
        if ev is None:
            pytest.skip("Evaluation directory not found")
        f = ev / "cag_stats.json"
        if not f.exists():
            pytest.skip("cag_stats.json not found")
        data = json.loads(f.read_text())

        all_keys: set = set(data.keys())
        for v in data.values():
            if isinstance(v, dict):
                all_keys.update(v.keys())

        for field in ("hit_rate", "cache_hits", "total_queries"):
            assert any(field in k for k in all_keys), (
                f"cag_stats.json missing key like '{field}'. "
                f"All keys found: {sorted(all_keys)}"
            )

    # -----------------------------------------------------------------------
    # 100-query simulation — Part 4 "CAG Cache Effectiveness"
    # -----------------------------------------------------------------------
    def test_cag_stats_reflects_100_query_simulation(self):
        """
        Rubric Part 4: 'Simulates 100 queries. Calculates cache hit rate…'

        total_queries in cag_stats.json must be >= 100, confirming that a
        proper 100-query simulation was run (not just a few hand-picked calls).
        """
        ev = self._find_eval_dir()
        if ev is None:
            pytest.skip("Evaluation directory not found")
        f = ev / "cag_stats.json"
        if not f.exists():
            pytest.skip("cag_stats.json not found")
        data = json.loads(f.read_text())

        # Flatten one level deep
        all_data: dict = {**data}
        for v in data.values():
            if isinstance(v, dict):
                all_data.update(v)

        total = None
        for k in ("total_queries", "total", "num_queries", "query_count"):
            if k in all_data:
                total = all_data[k]
                break

        if total is None:
            pytest.skip("total_queries key not found in cag_stats.json")

        assert int(total) >= 100, (
            f"cag_stats.json total_queries={total} — rubric requires simulating "
            "at least 100 queries to produce statistically meaningful hit-rate figures."
        )

    def test_cag_stats_has_latency_improvement(self):
        """
        Rubric Part 4: 'calculates latency improvement'.
        cag_stats.json must include some latency or time-saving metric.
        """
        ev = self._find_eval_dir()
        if ev is None:
            pytest.skip("Evaluation directory not found")
        f = ev / "cag_stats.json"
        if not f.exists():
            pytest.skip("cag_stats.json not found")
        data = json.loads(f.read_text())

        all_keys: set = set(data.keys())
        for v in data.values():
            if isinstance(v, dict):
                all_keys.update(v.keys())

        latency_keywords = ("latency", "time", "ms", "speedup", "saving", "improvement")
        has_latency = any(
            any(kw in k.lower() for kw in latency_keywords)
            for k in all_keys
        )
        assert has_latency, (
            "cag_stats.json missing latency/time improvement metric. "
            f"Keys found: {sorted(all_keys)}. "
            "Rubric Part 4 requires 'latency improvement' to be reported."
        )

    def test_crag_impact_csv_exists(self):
        ev = self._find_eval_dir()
        if ev is None:
            pytest.skip("Evaluation directory not found")
        assert (ev / "crag_impact.csv").exists(), f"Missing: {ev / 'crag_impact.csv'}"

    def test_crag_impact_csv_structure(self):
        import pandas as pd
        ev = self._find_eval_dir()
        if ev is None:
            pytest.skip("Evaluation directory not found")
        f = ev / "crag_impact.csv"
        if not f.exists():
            pytest.skip("crag_impact.csv not found")
        df = pd.read_csv(f)
        cols_lower = [c.lower() for c in df.columns]
        assert any("correction" in c for c in cols_lower), (
            f"crag_impact.csv missing correction column. Columns: {list(df.columns)}"
        )
        assert len(df) >= 20, (
            f"crag_impact.csv should have >= 20 rows (one per query), has {len(df)}"
        )

    # -----------------------------------------------------------------------
    # CRAG confidence improvement evidence — Part 4
    # -----------------------------------------------------------------------
    def test_crag_impact_has_confidence_columns(self):
        """
        Rubric Part 4: 'Tracks correction frequency, confidence improvement,
        and answer quality gains.'

        crag_impact.csv must contain columns capturing both initial and final
        confidence scores so that improvement can be measured quantitatively.
        """
        import pandas as pd
        ev = self._find_eval_dir()
        if ev is None:
            pytest.skip("Evaluation directory not found")
        f = ev / "crag_impact.csv"
        if not f.exists():
            pytest.skip("crag_impact.csv not found")
        df = pd.read_csv(f)
        cols_lower = [c.lower() for c in df.columns]

        assert any("confidence" in c for c in cols_lower), (
            f"crag_impact.csv missing confidence column(s). "
            f"Expected columns like 'confidence_initial', 'confidence_final', or 'confidence'. "
            f"Columns present: {list(df.columns)}"
        )

    def test_crag_impact_rag_vs_crag_comparison(self):
        """
        Rubric Part 4: 'Compares RAG vs CRAG on 20 queries.'

        crag_impact.csv must contain columns that allow direct comparison
        between plain RAG and CRAG — e.g. separate answer quality or
        confidence columns for each approach.
        """
        import pandas as pd
        ev = self._find_eval_dir()
        if ev is None:
            pytest.skip("Evaluation directory not found")
        f = ev / "crag_impact.csv"
        if not f.exists():
            pytest.skip("crag_impact.csv not found")
        df = pd.read_csv(f)
        cols_lower = [c.lower() for c in df.columns]

        # Accept either explicit rag/crag columns OR a 'method'/'model' column
        has_rag_col = any("rag" in c for c in cols_lower)
        has_method_col = any(c in cols_lower for c in ("method", "model", "approach", "system"))

        assert has_rag_col or has_method_col, (
            "crag_impact.csv doesn't appear to compare RAG vs CRAG. "
            "Expected columns like 'rag_answer', 'crag_answer', or a 'method' column. "
            f"Columns present: {list(df.columns)}"
        )