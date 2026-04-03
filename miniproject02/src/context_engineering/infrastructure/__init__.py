"""
Infrastructure layer for the Prime Lands Real Estate Intelligence Platform.

Acts as the integration boundary between the domain/application layers and
all external services.  Exposes the most commonly used infrastructure
primitives directly so higher-level code avoids deep import paths.

Modules:
    llm_providers: Unified factory functions for chat LLMs (``get_chat_llm``)
        and embedding models (``get_default_embeddings``) across all supported
        providers (OpenAI, Groq, Gemini, DeepSeek, OpenRouter, Cohere,
        HuggingFace).
    db:         Database and vector-store integrations (Qdrant persistence,
        collection management).
    api:        External API endpoint clients and request helpers.
    monitoring: Logging configuration, metrics collection, and tracing.
"""

from .llm_providers import get_chat_llm, get_default_embeddings

__all__ = [
    "get_chat_llm",
    "get_default_embeddings",
]