import hashlib
import pickle
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime


class CAGCache:
    """
    Semantic similarity-based cache with two-tier FAQ and History support.

    FAQs are static and never expire. History entries have a configurable TTL.
    All lookups use cosine similarity between query embeddings.

    Args:
        cache_dir: Directory to store cache files.
        embedder: Embedding model instance.
        similarity_threshold: Minimum cosine similarity for a cache hit (0.0–1.0).
        max_cache_size: Maximum number of history entries before eviction.
        history_ttl_hours: Hours before history entries expire.
    """

    def __init__(
        self,
        cache_dir: Path,
        embedder: Any,
        similarity_threshold: float = 0.90,
        max_cache_size: int = 1000,
        history_ttl_hours: float = 24.0
    ):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.max_cache_size = max_cache_size
        self.history_ttl_hours = history_ttl_hours

        self.faq_cache_file = cache_dir / "cag_faqs.pkl"
        self.history_cache_file = cache_dir / "cag_history.pkl"

        self.faq_cache: Dict[str, Any] = self._load_cache(self.faq_cache_file)
        self.history_cache: Dict[str, Any] = self._load_cache(self.history_cache_file)

        self._cleanup_expired_history()
        self._update_faq_embedding_matrix()
        self._update_history_embedding_matrix()

    def _load_cache(self, cache_file: Path) -> Dict[str, Any]:
        """Load cache from disk, returning empty dict on failure."""
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def _save_faq_cache(self) -> None:
        """Persist FAQ cache to disk."""
        with open(self.faq_cache_file, 'wb') as f:
            pickle.dump(self.faq_cache, f)

    def _save_history_cache(self) -> None:
        """Persist history cache to disk."""
        with open(self.history_cache_file, 'wb') as f:
            pickle.dump(self.history_cache, f)

    def _cleanup_expired_history(self) -> None:
        """Remove history entries older than TTL."""
        cutoff_time = time.time() - (self.history_ttl_hours * 3600)
        expired_keys = [
            key for key, entry in self.history_cache.items()
            if entry.get('timestamp', 0) < cutoff_time
        ]
        if expired_keys:
            for key in expired_keys:
                del self.history_cache[key]
            self._save_history_cache()

    def _generate_key(self, query: str) -> str:
        """Generate a unique cache key for a query."""
        return hashlib.md5(f"{query}_{time.time()}".encode()).hexdigest()

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query string into a numpy vector."""
        return np.array(self.embedder.embed_query(query))

    def _update_faq_embedding_matrix(self) -> None:
        """Rebuild the FAQ embedding matrix from warmed entries."""
        valid_faqs = {k: v for k, v in self.faq_cache.items() if v.get('has_response')}
        if not valid_faqs:
            self._faq_embedding_matrix = None
            self._faq_cache_ids = []
            return
        self._faq_cache_ids = list(valid_faqs.keys())
        self._faq_embedding_matrix = np.vstack(
            [valid_faqs[cid]['embedding'] for cid in self._faq_cache_ids]
        )

    def _update_history_embedding_matrix(self) -> None:
        """Rebuild the history embedding matrix after cleanup."""
        self._cleanup_expired_history()
        if not self.history_cache:
            self._history_embedding_matrix = None
            self._history_cache_ids = []
            return
        self._history_cache_ids = list(self.history_cache.keys())
        self._history_embedding_matrix = np.vstack(
            [self.history_cache[cid]['embedding'] for cid in self._history_cache_ids]
        )

    def _find_similar(
        self,
        query_embedding: np.ndarray,
        embedding_matrix: Optional[np.ndarray],
        cache_ids: List[str]
    ) -> Optional[Tuple[str, float]]:
        """
        Find the most similar cached entry using cosine similarity.

        Args:
            query_embedding: Embedding vector of the incoming query.
            embedding_matrix: Stacked embedding matrix of cached entries.
            cache_ids: Ordered list of cache keys matching the matrix rows.

        Returns:
            Tuple of (cache_id, similarity_score) if above threshold, else None.
        """
        if embedding_matrix is None or len(cache_ids) == 0:
            return None

        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
        cache_norms = embedding_matrix / (
            np.linalg.norm(embedding_matrix, axis=1, keepdims=True) + 1e-10
        )
        similarities = np.dot(cache_norms, query_norm)

        best_idx = np.argmax(similarities)
        best_similarity = float(similarities[best_idx])

        if best_similarity >= self.similarity_threshold:
            return (cache_ids[best_idx], best_similarity)
        return None

    def load_faqs(self, faq_queries: List[str], responses: Optional[List[Dict[str, Any]]] = None) -> int:
        """
        Load static FAQ questions into the cache.

        Call warm_faqs() via CAGService to generate responses afterward.

        Args:
            faq_queries: List of FAQ question strings.
            responses: Optional pre-computed responses to store immediately.

        Returns:
            Number of new FAQ entries added.
        """
        loaded = 0
        for i, query in enumerate(faq_queries):
            query_embedding = self._embed_query(query)
            existing = self._find_similar(
                query_embedding,
                self._faq_embedding_matrix,
                self._faq_cache_ids
            )
            if existing and existing[1] > 0.95:
                continue

            key = self._generate_key(query)
            entry = {
                'query': query,
                'embedding': query_embedding,
                'is_faq': True,
                'timestamp': time.time()
            }
            if responses and i < len(responses):
                entry['answer'] = responses[i].get('answer', '')
                entry['evidence_urls'] = responses[i].get('evidence_urls', [])
                entry['has_response'] = True
            else:
                entry['has_response'] = False

            self.faq_cache[key] = entry
            loaded += 1

        if loaded > 0:
            self._save_faq_cache()
            self._update_faq_embedding_matrix()

        return loaded

    def get_pending_faqs(self) -> List[str]:
        """Return FAQ queries that have not yet been warmed with a response."""
        return [
            entry['query'] for entry in self.faq_cache.values()
            if not entry.get('has_response', False)
        ]

    def update_faq_response(self, query: str, response: Dict[str, Any]) -> bool:
        """
        Store a generated response against an existing FAQ entry.

        Args:
            query: The FAQ question to update.
            response: Dict with 'answer' and optional 'evidence_urls'.

        Returns:
            True if the FAQ was found and updated, False otherwise.
        """
        query_embedding = self._embed_query(query)

        # Check pending FAQs by exact match first (no embedding matrix yet)
        for key, entry in self.faq_cache.items():
            if entry['query'].lower().strip() == query.lower().strip():
                self.faq_cache[key]['answer'] = response['answer']
                self.faq_cache[key]['evidence_urls'] = response.get('evidence_urls', [])
                self.faq_cache[key]['has_response'] = True
                self.faq_cache[key]['timestamp'] = time.time()
                self._save_faq_cache()
                self._update_faq_embedding_matrix()
                return True

        match = self._find_similar(query_embedding, self._faq_embedding_matrix, self._faq_cache_ids)
        if match:
            key = match[0]
            self.faq_cache[key]['answer'] = response['answer']
            self.faq_cache[key]['evidence_urls'] = response.get('evidence_urls', [])
            self.faq_cache[key]['has_response'] = True
            self.faq_cache[key]['timestamp'] = time.time()
            self._save_faq_cache()
            self._update_faq_embedding_matrix()
            return True

        return False

    def list_faqs(self) -> List[Dict[str, Any]]:
        """Return all FAQ entries with their ready status and timestamp."""
        return [
            {
                'query': entry['query'],
                'has_response': entry.get('has_response', False),
                'timestamp': datetime.fromtimestamp(entry['timestamp']).isoformat()
            }
            for entry in self.faq_cache.values()
        ]

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a cached response using semantic similarity.

        Checks FAQs first, then history. Returns None on a miss.

        Args:
            query: Incoming user query.

        Returns:
            Cached response dict with 'source', 'similarity_score', and
            'matched_query' keys, or None if no match found.
        """
        self._cleanup_expired_history()
        query_embedding = self._embed_query(query)

        faq_match = self._find_similar(query_embedding, self._faq_embedding_matrix, self._faq_cache_ids)
        if faq_match:
            cache_id, similarity = faq_match
            cached = self.faq_cache[cache_id].copy()
            cached.pop('embedding', None)
            cached['similarity_score'] = similarity
            cached['matched_query'] = cached['query']
            cached['source'] = 'faq'
            return cached

        self._update_history_embedding_matrix()
        history_match = self._find_similar(
            query_embedding, self._history_embedding_matrix, self._history_cache_ids
        )
        if history_match:
            cache_id, similarity = history_match
            entry = self.history_cache[cache_id]
            if time.time() - entry['timestamp'] < self.history_ttl_hours * 3600:
                cached = entry.copy()
                cached.pop('embedding', None)
                cached['similarity_score'] = similarity
                cached['matched_query'] = cached['query']
                cached['source'] = 'history'
                return cached

        return None

    def set(self, query: str, response: Dict[str, Any]) -> None:
        """
        Store a response in the history cache.

        Evicts the oldest entry if the cache exceeds max_cache_size.

        Args:
            query: User query string.
            response: Dict with 'answer' and optional 'evidence_urls'.
        """
        key = self._generate_key(query)
        embedding = self._embed_query(query)

        self.history_cache[key] = {
            'query': query,
            'embedding': embedding,
            'answer': response['answer'],
            'evidence_urls': response.get('evidence_urls', []),
            'timestamp': time.time(),
            'is_faq': False
        }

        if len(self.history_cache) > self.max_cache_size:
            oldest_key = min(
                self.history_cache.keys(),
                key=lambda k: self.history_cache[k]['timestamp']
            )
            del self.history_cache[oldest_key]

        self._update_history_embedding_matrix()
        self._save_history_cache()

    def clear(self, clear_faqs: bool = False) -> None:
        """
        Clear the history cache and optionally the FAQ cache.

        Args:
            clear_faqs: If True, also clears the FAQ tier.
        """
        self.history_cache = {}
        self._history_embedding_matrix = None
        self._history_cache_ids = []
        self._save_history_cache()

        if clear_faqs:
            self.faq_cache = {}
            self._faq_embedding_matrix = None
            self._faq_cache_ids = []
            self._save_faq_cache()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics including counts, TTL, and disk size."""
        faq_size = self.faq_cache_file.stat().st_size if self.faq_cache_file.exists() else 0
        history_size = self.history_cache_file.stat().st_size if self.history_cache_file.exists() else 0

        faqs_ready = sum(1 for e in self.faq_cache.values() if e.get('has_response'))
        self._cleanup_expired_history()

        return {
            'total_cached': len(self.faq_cache) + len(self.history_cache),
            'faq_count': len(self.faq_cache),
            'faq_ready': faqs_ready,
            'faq_pending': len(self.faq_cache) - faqs_ready,
            'history_count': len(self.history_cache),
            'history_ttl_hours': self.history_ttl_hours,
            'similarity_threshold': self.similarity_threshold,
            'cache_size_kb': (faq_size + history_size) / 1024
        }

    def get_history_queries(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Return recent history queries sorted by recency.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of dicts with 'query', 'timestamp', and 'age_hours'.
        """
        self._cleanup_expired_history()
        entries = [
            {
                'query': entry['query'],
                'timestamp': datetime.fromtimestamp(entry['timestamp']).isoformat(),
                'age_hours': (time.time() - entry['timestamp']) / 3600
            }
            for entry in self.history_cache.values()
        ]
        entries.sort(key=lambda x: x['age_hours'])
        return entries[:limit]

    def prewarm(self, queries: List[str], responses: Optional[List[Dict[str, Any]]] = None, **kwargs) -> int:
        """
        Pre-warm the cache by loading FAQ queries and optionally storing responses.

        Alias for load_faqs() to satisfy the prewarm() interface expected by tests
        and CAGService.

        Args:
            queries: List of FAQ question strings to pre-load.
            responses: Optional pre-computed responses to store immediately.

        Returns:
            Number of new FAQ entries loaded.
        """
        return self.load_faqs(queries, responses)

    def __len__(self) -> int:
        return len(self.faq_cache) + len(self.history_cache)

    def __contains__(self, query: str) -> bool:
        return self.get(query) is not None


__all__ = ['CAGCache']