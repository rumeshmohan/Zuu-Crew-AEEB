"""
Helper utilities for the Prime Lands RAG pipeline.

Provides document formatting, confidence scoring, citation extraction,
text truncation, and token counting used across RAGService, CAGService,
and CRAGService.
"""

import re
from typing import List

import tiktoken


def format_docs(docs: list) -> str:
    """Format a list of Documents into a single numbered context string.

    Each entry includes the source URL, page title, and a content preview
    (first 500 characters).  Entries are separated by a horizontal rule so
    the LLM can distinguish individual sources.

    Args:
        docs (list): LangChain ``Document`` objects returned by a retriever.

    Returns:
        str: Multi-source context block ready for prompt injection.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        url = doc.metadata.get('url', 'N/A')
        title = doc.metadata.get('title', 'N/A')
        content = doc.page_content[:500]
        formatted.append(
            f"[Source {i}: {url}]\n"
            f"Title: {title}\n"
            f"Content: {content}\n"
        )
    return "\n---\n".join(formatted)


def calculate_confidence(docs: list, query: str) -> float:
    """Compute a heuristic confidence score for a set of retrieved documents.

    Combines three weighted factors:

    * **Keyword overlap** (weight 0.5) – fraction of query tokens present in
      each document, averaged across all docs.
    * **Content richness** (weight 0.3) – average document length normalised
      to a 500-character ceiling.
    * **Strategy diversity** (weight 0.2) – number of distinct chunking
      strategies represented, normalised to a maximum of 3.

    Args:
        docs (list): Retrieved ``Document`` objects to evaluate.
        query (str): Original user query string.

    Returns:
        float: Confidence score in the range ``[0.0, 1.0]``.
    """
    if not docs:
        return 0.0

    query_words = set(query.lower().split())

    # Factor 1: keyword overlap between query and each document
    overlaps = []
    for doc in docs:
        doc_words = set(doc.page_content.lower().split())
        overlap = len(query_words & doc_words) / len(query_words) if query_words else 0
        overlaps.append(overlap)
    keyword_score = max(overlaps)

    # Factor 2: content richness — longer docs generally carry more signal
    avg_length = sum(len(doc.page_content) for doc in docs) / len(docs)
    length_score = min(avg_length / 500, 1.0)

    # Factor 3: URL source diversity (weight 0.2)
    # More docs from different pages = more evidence breadth
    unique_urls = set(doc.metadata.get('url', 'unknown') for doc in docs)
    diversity_score = min(len(unique_urls) / len(docs), 1.0)

    confidence = (
        0.5 * keyword_score +
        0.3 * length_score +
        0.2 * diversity_score
    )

    return confidence


def extract_citations(text: str) -> List[str]:
    """Extract bracketed URL citations from LLM-generated answer text.

    Scans for ``[…]`` patterns and retains only entries that look like URLs
    (containing ``http`` or ``.com``).

    Args:
        text (str): Generated answer that may contain inline ``[url]`` citations.

    Returns:
        List[str]: Unique URLs found in the text, in order of appearance.
    """
    # Match everything inside square brackets
    citations = re.findall(r'\[([^\]]+)\]', text)

    # Keep only entries that resemble URLs
    urls = [c for c in citations if 'http' in c or '.com' in c]

    return urls


def truncate_text(text: str, max_length: int = 400) -> str:
    """Truncate text to a maximum character length without splitting words.

    If the text exceeds ``max_length``, it is cut at the last word boundary
    before the limit and an ellipsis is appended.

    Args:
        text (str): Input text to shorten.
        max_length (int): Maximum number of characters to retain. Defaults to 400.

    Returns:
        str: Original text if within limit, otherwise a word-boundary truncation
        followed by ``"…"``.
    """
    if len(text) <= max_length:
        return text

    return text[:max_length].rsplit(' ', 1)[0] + "..."


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count the number of tokens in a text string using tiktoken.

    Attempts to load the tokeniser for the specified model; falls back to the
    ``cl100k_base`` encoding if the model name is not recognised by tiktoken.

    Args:
        text (str): The text to tokenise and count.
        model (str): Model name used to select the correct tokeniser.
            Defaults to ``"gpt-3.5-turbo"``.

    Returns:
        int: Token count for ``text`` under the chosen encoding.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to cl100k_base when the model is not in tiktoken's registry
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))