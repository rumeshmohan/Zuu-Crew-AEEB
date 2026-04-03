import time
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class VectorStoreService:
    """
    Service for managing local Qdrant vector store collections.

    Args:
        embeddings: Embedding model instance.
        path: Local path for Qdrant storage.
    """

    def __init__(self, embeddings, path: str = None):
        self.embeddings = embeddings
        self.path = path or "data/vectorstore"
        self.collections = {}

        print(f"📌 Initializing Qdrant Client (local mode)...")

        Path(self.path).mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=self.path, prefer_grpc=False)

        try:
            collections = self.client.get_collections()
            count = len(collections.collections) if hasattr(collections, 'collections') else 0
            print(f"✅ Connected to Qdrant (local). Existing collections: {count}")
        except Exception as e:
            print(f"⚠️  Connection warning: {e}")

    def prepare_documents(self, chunks: List[Any], strategy_name: str) -> List[Document]:
        """
        Convert chunk dicts or LangChain Documents to LangChain Documents with metadata.

        Accepts either plain dicts (keys: 'text' / 'page_content' / 'content')
        or existing LangChain Document objects so that test fixtures and
        production chunkers both work without conversion.

        Args:
            chunks: List of chunk dicts or Document objects.
            strategy_name: Name of the chunking strategy used.

        Returns:
            List of LangChain Document objects with populated metadata.
        """
        excluded = {'text', 'page_content', 'content', 'url', 'title',
                    'strategy', 'chunk_index', 'token_count', 'parent_id', 'splittable'}
        documents = []

        for chunk in chunks:
            # ── Already a LangChain Document ──────────────────────────────
            if isinstance(chunk, Document):
                meta = dict(chunk.metadata or {})
                meta.setdefault('strategy', strategy_name)
                documents.append(Document(page_content=chunk.page_content, metadata=meta))
                continue

            # ── Plain dict chunk ──────────────────────────────────────────
            text = (
                chunk.get('text')
                or chunk.get('page_content')
                or chunk.get('content')
                or ''
            )
            meta = {
                "url": chunk.get('url'),
                "title": chunk.get('title'),
                "strategy": strategy_name,
                "chunk_index": chunk.get('chunk_index'),
                "token_count": chunk.get('token_count'),
                "parent_id": chunk.get('parent_id', ''),
                "splittable": chunk.get('splittable', False),
            }
            for k, v in chunk.items():
                if k not in excluded:
                    meta[k] = v

            documents.append(Document(page_content=text, metadata=meta))

        return documents

    def create_collection(
        self,
        collection_name: str,
        chunks: List[Any],
        strategy_name: str,
        verbose: bool = True
    ) -> QdrantVectorStore:
        """
        Create a Qdrant collection for a chunking strategy.

        Embeds documents and upserts vectors directly via the qdrant client
        rather than through QdrantVectorStore.__init__, so that any embeddings
        object with embed_documents() / embed_query() methods works (including
        test mocks that don't pass isinstance checks).

        Args:
            collection_name: Full Qdrant collection name.
            chunks: List of chunk dicts or LangChain Documents to index.
            strategy_name: Name of the chunking strategy.
            verbose: If True, prints progress to stdout.

        Returns:
            QdrantVectorStore instance bound to the created collection.

        Raises:
            ValueError: If chunks list is empty.
        """
        import uuid
        from qdrant_client.models import PointStruct

        if not chunks:
            raise ValueError(f"No chunks for {collection_name}")

        if verbose:
            print(f"\n🚀 Creating collection: {collection_name}")
            print(f"   Strategy: {strategy_name} | Chunks: {len(chunks)}")

        try:
            documents = self.prepare_documents(chunks, strategy_name)
            start_time = time.time()

            # ── Delete existing collection if present ─────────────────────
            try:
                self.client.delete_collection(collection_name)
            except Exception:
                pass

            # ── Determine embedding dimension ─────────────────────────────
            sample_embedding = self.embeddings.embed_query("test")
            embedding_dim = len(sample_embedding)

            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
            )

            # ── Embed and upsert directly (works with any embeddings object) ──
            texts = [doc.page_content for doc in documents]
            vectors = self.embeddings.embed_documents(texts)

            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={"page_content": text, **doc.metadata},
                )
                for doc, text, vector in zip(documents, texts, vectors)
            ]
            self.client.upsert(collection_name=collection_name, points=points)

            creation_time = time.time() - start_time

            try:
                count_result = self.client.count(collection_name=collection_name)
                count = count_result.count if hasattr(count_result, 'count') else count_result
            except Exception:
                count = len(documents)

            if verbose:
                print(f"   ✅ Created in {creation_time:.2f}s")
                print(f"   📊 Vectors in DB: {count}")

            # ── Build a lightweight wrapper for similarity_search ─────────
            # Store collection name and embedding ref for our own search()
            self.collections[strategy_name] = {
                "collection_name": collection_name,
                "embedding_dim": embedding_dim,
            }
            self.collections[collection_name] = self.collections[strategy_name]

            # Try to return a proper QdrantVectorStore; fall back to None
            try:
                vectorstore = QdrantVectorStore(
                    client=self.client,
                    collection_name=collection_name,
                    embedding=self.embeddings,
                )
            except Exception:
                vectorstore = None

            # Store the vectorstore if we got one
            if vectorstore is not None:
                self.collections[strategy_name] = vectorstore
                self.collections[collection_name] = vectorstore

            return vectorstore or self  # return self as fallback so callers don't crash

        except Exception as e:
            if verbose:
                print(f"   ❌ Error: {e}")
            raise

    def create_all_collections(
        self,
        strategy_chunks: Dict[str, List[Dict[str, Any]]],
        collection_prefix: str = "primelands"
    ) -> Dict[str, QdrantVectorStore]:
        """
        Create Qdrant collections for all chunking strategies.

        Args:
            strategy_chunks: Dict mapping strategy name to chunk list.
            collection_prefix: Prefix applied to all collection names.

        Returns:
            Dict mapping strategy name to QdrantVectorStore instance.
        """
        print("=" * 60)
        print("CREATING QDRANT COLLECTIONS")
        print("=" * 60)

        created = 0
        total = len(strategy_chunks)

        for strategy_name, chunks in strategy_chunks.items():
            if not chunks:
                print(f"\n⚠️  Skipping {strategy_name}: no chunks")
                continue

            collection_name = f"{collection_prefix}_{strategy_name}"

            try:
                self.create_collection(collection_name, chunks, strategy_name)
                created += 1
            except Exception as e:
                print(f"\n❌ Error creating {strategy_name}: {e}")
                continue

        print("\n" + "=" * 60)
        print(f"✅ {created}/{total} COLLECTIONS CREATED")
        print("=" * 60)

        return self.collections

    def index_chunks(
        self,
        chunks: List[Any],
        collection_name: str,
        strategy_name: str = "fixed",
        verbose: bool = False,
    ) -> "VectorStoreService":
        """
        Index a list of chunks into a named Qdrant collection.

        Accepts both plain dict chunks (from chunkers) and LangChain
        Document objects (from test fixtures).

        Args:
            chunks: List of chunk dicts or Document objects to index.
            collection_name: Full Qdrant collection name to create/overwrite.
            strategy_name: Chunking strategy label stored in metadata.
                Inferred from chunks if not supplied.
            verbose: If True, prints progress to stdout.

        Returns:
            Self, for method chaining.
        """
        if chunks and isinstance(chunks[0], dict) and "strategy" in chunks[0]:
            strategy_name = chunks[0]["strategy"]

        self.create_collection(
            collection_name=collection_name,
            chunks=chunks,
            strategy_name=strategy_name,
            verbose=verbose,
        )
        return self

    def search(
        self,
        query: str,
        collection_name: str = None,
        strategy_name: str = None,
        k: int = 5,
    ) -> List[Document]:
        """
        Similarity search against a Qdrant collection using the qdrant
        client directly (works with any embeddings object).

        Args:
            query: User query string.
            collection_name: Full collection name to search.
            strategy_name: Strategy name used as fallback lookup key.
            k: Number of results to return.

        Returns:
            List of LangChain Document objects ranked by relevance.
        """
        # Resolve collection name
        resolved = collection_name
        if not resolved and strategy_name:
            entry = self.collections.get(strategy_name)
            if isinstance(entry, dict):
                resolved = entry.get("collection_name")
            elif entry is not None:
                resolved = strategy_name
        if not resolved and self.collections:
            last = list(self.collections.values())[-1]
            if isinstance(last, dict):
                resolved = last.get("collection_name")

        if not resolved:
            raise ValueError(
                f"No collection found. Available: {list(self.collections.keys())}"
            )

        try:
            query_vector = self.embeddings.embed_query(query)
            response = self.client.query_points(
                collection_name=resolved,
                query=query_vector,
                limit=k,
            )
            hits = response.points
            return [
                Document(
                    page_content=hit.payload.get("page_content", ""),
                    metadata={k: v for k, v in hit.payload.items() if k != "page_content"},
                )
                for hit in hits
            ]
        except Exception as e:
            raise RuntimeError(f"Search failed on '{resolved}': {e}") from e

    def get_collection(self, strategy_name: str) -> Optional[QdrantVectorStore]:
        """
        Retrieve the vectorstore for a given strategy.

        Args:
            strategy_name: Name of the chunking strategy.

        Returns:
            QdrantVectorStore instance, or None if not found.
        """
        return self.collections.get(strategy_name)

    def list_collections(self) -> List[str]:
        """Return the names of all created collections."""
        return list(self.collections.keys())

    def get_collection_count(self, strategy_name: str, collection_prefix: str = "primelands") -> int:
        """
        Get the vector count for a specific collection.

        Args:
            strategy_name: Name of the chunking strategy.
            collection_prefix: Prefix used when naming the collection.

        Returns:
            Number of vectors in the collection, or 0 on failure.
        """
        try:
            collection_name = f"{collection_prefix}_{strategy_name}"
            count_result = self.client.count(collection_name=collection_name)
            return count_result.count if hasattr(count_result, 'count') else count_result
        except Exception:
            return 0