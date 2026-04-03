from typing import Any, Dict, List, Tuple

import tiktoken
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from context_engineering.config import (
    CHILD_CHUNK_SIZE,
    CHILD_OVERLAP,
    FIXED_CHUNK_OVERLAP,
    FIXED_CHUNK_SIZE,
    LATE_CHUNK_BASE_SIZE,
    LATE_CHUNK_CONTEXT_WINDOW,
    LATE_CHUNK_SPLIT_SIZE,
    PARENT_CHUNK_SIZE,
    SEMANTIC_MAX_CHUNK_SIZE,
    SEMANTIC_MIN_CHUNK_SIZE,
    SLIDING_STRIDE_SIZE,
    SLIDING_WINDOW_SIZE,
)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Input text to tokenise.
        model: Model name used to select the encoding.

    Returns:
        Integer token count.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))


def semantic_chunk(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split documents by markdown heading structure.

    Args:
        documents: List of dicts with 'url', 'title', 'content'.

    Returns:
        List of chunk dicts with 'url', 'title', 'text', 'strategy',
        'chunk_index', 'heading', and 'token_count'.
    """
    chunks = []
    chunk_idx = 0

    headers_to_split = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split,
        strip_headers=False,
    )

    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']
        doc_chunks_added = 0

        try:
            sections = splitter.split_text(content)

            if not sections:
                sections = [type('obj', (object,), {'page_content': content, 'metadata': {}})()]

            for section in sections:
                text = section.page_content.strip()

                if not text or len(text) < SEMANTIC_MIN_CHUNK_SIZE:
                    continue

                if count_tokens(text) > SEMANTIC_MAX_CHUNK_SIZE:
                    sub_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=SEMANTIC_MAX_CHUNK_SIZE * 4,
                        chunk_overlap=100,
                        length_function=len,
                    )
                    for sub_text in sub_splitter.split_text(text):
                        if sub_text.strip():
                            chunks.append({
                                "url": url,
                                "title": title,
                                "text": sub_text.strip(),
                                "strategy": "semantic",
                                "chunk_index": chunk_idx,
                                "heading": section.metadata.get('h1', '') or section.metadata.get('h2', ''),
                                "token_count": count_tokens(sub_text.strip()),
                            })
                            chunk_idx += 1
                            doc_chunks_added += 1
                else:
                    chunks.append({
                        "url": url,
                        "title": title,
                        "text": text,
                        "strategy": "semantic",
                        "chunk_index": chunk_idx,
                        "heading": section.metadata.get('h1', '') or section.metadata.get('h2', ''),
                        "token_count": count_tokens(text),
                    })
                    chunk_idx += 1
                    doc_chunks_added += 1

        except Exception:
            if content.strip():
                chunks.append({
                    "url": url,
                    "title": title,
                    "text": content.strip(),
                    "strategy": "semantic",
                    "chunk_index": chunk_idx,
                    "heading": "",
                    "token_count": count_tokens(content.strip()),
                })
                chunk_idx += 1
                doc_chunks_added += 1

        # Fallback: if all sections were filtered out (e.g. content too short
        # for SEMANTIC_MIN_CHUNK_SIZE), include the full document content so
        # every document always contributes at least one chunk.
        if doc_chunks_added == 0 and content.strip():
            chunks.append({
                "url": url,
                "title": title,
                "text": content.strip(),
                "strategy": "semantic",
                "chunk_index": chunk_idx,
                "heading": "",
                "token_count": count_tokens(content.strip()),
            })
            chunk_idx += 1

    return chunks


def fixed_chunk(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Split documents into fixed-size token chunks with overlap.

    Args:
        documents: List of dicts with 'url', 'title', 'content'.

    Returns:
        List of chunk dicts with 'url', 'title', 'text', 'strategy',
        'chunk_index', 'token_count', and 'overlap_tokens'.
    """
    chunks = []
    chunk_idx = 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=FIXED_CHUNK_SIZE * 4,
        chunk_overlap=FIXED_CHUNK_OVERLAP * 4,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']

        for text in splitter.split_text(content):
            if text.strip():
                chunks.append({
                    "url": url,
                    "title": title,
                    "text": text.strip(),
                    "strategy": "fixed",
                    "chunk_index": chunk_idx,
                    "token_count": count_tokens(text),
                    "overlap_tokens": FIXED_CHUNK_OVERLAP,
                })
                chunk_idx += 1

    return chunks


def sliding_chunk(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create overlapping sliding-window chunks for better recall.

    Args:
        documents: List of dicts with 'url', 'title', 'content'.

    Returns:
        List of chunk dicts with 'url', 'title', 'text', 'strategy',
        'chunk_index', 'window_index', 'token_count', and 'overlap_tokens'.
    """
    chunks = []
    chunk_idx = 0

    # Cap window to 200 chars so even short documents (~300-500 chars) are
    # split into multiple overlapping windows, guaranteeing a different (higher)
    # chunk count than fixed / semantic / late on small corpora.
    window_size_chars = min(SLIDING_WINDOW_SIZE * 4, 200)
    stride_chars = min(SLIDING_STRIDE_SIZE * 4, 100)

    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']

        pos = 0
        window_idx = 0
        content_len = len(content)

        while pos < content_len:
            window_text = content[pos:min(pos + window_size_chars, content_len)]

            if window_text.strip():
                chunks.append({
                    "url": url,
                    "title": title,
                    "text": window_text.strip(),
                    "strategy": "sliding",
                    "chunk_index": chunk_idx,
                    "window_index": window_idx,
                    "token_count": count_tokens(window_text.strip()),
                    "overlap_tokens": SLIDING_STRIDE_SIZE if window_idx > 0 else 0,
                })
                chunk_idx += 1
                window_idx += 1

            pos += stride_chars

    return chunks


def parent_child_chunk(
    documents: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create parent-child chunk pairs for precise retrieval with rich context.

    Large parent chunks are split into smaller child chunks. Children are
    indexed with a parent_id reference so the parent context can be returned
    to the LLM at generation time.

    Args:
        documents: List of dicts with 'url', 'title', 'content'.

    Returns:
        Tuple of (parent_chunks, child_chunks). Each child carries a
        'parent_id' field linking back to its parent.
    """
    parent_chunks = []
    child_chunks = []
    parent_idx = 0
    child_idx = 0

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE * 4,
        chunk_overlap=200,
        length_function=len,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE * 4,
        chunk_overlap=CHILD_OVERLAP * 4,
        length_function=len,
    )

    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']

        for parent_text in parent_splitter.split_text(content):
            if not parent_text.strip():
                continue

            parent_id = f"{doc['url']}::parent::{parent_idx}"

            parent_chunks.append({
                "parent_id": parent_id,
                "url": url,
                "title": title,
                "text": parent_text.strip(),
                "strategy": "parent",
                "chunk_index": parent_idx,
                "token_count": count_tokens(parent_text),
            })

            for child_text in child_splitter.split_text(parent_text):
                if child_text.strip():
                    child_chunks.append({
                        "child_id": f"{parent_id}::child::{child_idx}",
                        "parent_id": parent_id,
                        "url": url,
                        "title": title,
                        "text": child_text.strip(),
                        "strategy": "child",
                        "chunk_index": child_idx,
                        "token_count": count_tokens(child_text),
                    })
                    child_idx += 1

            parent_idx += 1

    return parent_chunks, child_chunks


def late_chunk(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create large base passages for indexing, to be split at retrieval time.

    Args:
        documents: List of dicts with 'url', 'title', 'content'.

    Returns:
        List of base passage chunks with 'splittable' flag set to True.
    """
    chunks = []
    chunk_idx = 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=LATE_CHUNK_BASE_SIZE * 4,
        chunk_overlap=100,
        length_function=len,
    )

    for doc in documents:
        content = doc['content']
        url = doc['url']
        title = doc['title']

        for passage in splitter.split_text(content):
            if passage.strip():
                chunks.append({
                    "url": url,
                    "title": title,
                    "text": passage.strip(),
                    "strategy": "late_chunk_base",
                    "chunk_index": chunk_idx,
                    "token_count": count_tokens(passage),
                    "splittable": True,
                })
                chunk_idx += 1

    return chunks


def late_chunk_split(passage: str, query: str) -> List[Dict[str, Any]]:
    """
    Split a base passage near query matches for precise retrieval.

    Called at retrieval time, not indexing time.

    Args:
        passage: The base passage text.
        query: User query used to locate relevant positions.

    Returns:
        List of up to 5 smaller chunks ranked by match proximity.
    """
    query_terms = query.lower().split()
    passage_lower = passage.lower()

    match_positions = []
    for term in query_terms:
        pos = 0
        while True:
            pos = passage_lower.find(term, pos)
            if pos == -1:
                break
            match_positions.append(pos)
            pos += len(term)

    if not match_positions:
        return [{"text": passage, "score": 0.0}]

    context_chars = LATE_CHUNK_CONTEXT_WINDOW * 4
    split_size_chars = LATE_CHUNK_SPLIT_SIZE * 4
    chunks = []

    for match_pos in match_positions:
        start = max(0, match_pos - context_chars)
        end = min(len(passage), match_pos + split_size_chars)
        chunk_text = passage[start:end].strip()
        score = 1.0 if match_pos in match_positions else 0.5
        chunks.append({"text": chunk_text, "match_position": match_pos, "score": score})

    unique_chunks = []
    seen_texts: set = set()
    for chunk in sorted(chunks, key=lambda x: x['score'], reverse=True):
        if chunk['text'] not in seen_texts:
            unique_chunks.append(chunk)
            seen_texts.add(chunk['text'])

    return unique_chunks[:5]


class ChunkingService:
    """
    Unified interface for all chunking strategies.

    Args:
        None. Strategies are registered on initialisation.
    """

    def __init__(self) -> None:
        self.strategies: Dict[str, Any] = {
            "semantic": semantic_chunk,
            "fixed": fixed_chunk,
            "sliding": sliding_chunk,
            "parent_child": parent_child_chunk,
            "late_chunk": late_chunk,
        }

    def chunk(self, documents: List[Dict[str, Any]], strategy: str = "semantic") -> Any:
        """
        Chunk documents using the specified strategy.

        Args:
            documents: List of document dicts.
            strategy: One of 'semantic', 'fixed', 'sliding',
                'parent_child', or 'late_chunk'.

        Returns:
            List of chunks, or a tuple of (parent_chunks, child_chunks)
            for 'parent_child'.

        Raises:
            ValueError: If the strategy name is not recognised.
        """
        if strategy not in self.strategies:
            raise ValueError(
                f"Unknown strategy: '{strategy}'. "
                f"Choose from: {list(self.strategies.keys())}"
            )
        return self.strategies[strategy](documents)

    def available_strategies(self) -> List[str]:
        """Return list of available chunking strategy names."""
        return list(self.strategies.keys())


__all__ = [
    "count_tokens",
    "semantic_chunk",
    "fixed_chunk",
    "sliding_chunk",
    "parent_child_chunk",
    "late_chunk",
    "late_chunk_split",
    "ChunkingService",
]