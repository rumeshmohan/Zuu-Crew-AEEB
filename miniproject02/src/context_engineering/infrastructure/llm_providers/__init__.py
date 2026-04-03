"""
LLM providers sub-package for the Prime Lands Real Estate Intelligence Platform.

Provides a unified import surface for all embedding and chat model factories,
consolidating access to OpenAI, HuggingFace, Cohere, Groq, DeepSeek,
OpenRouter, and Google Gemini providers configured via ``config.yaml``.

Modules:
    embeddings:    ``get_default_embeddings`` — provider-agnostic embedding
        factory; ``get_small_embeddings`` / ``get_large_embeddings`` —
        OpenAI size-specific convenience wrappers; ``detect_available_providers``
        / ``list_available_providers`` — runtime provider introspection;
        ``print_provider_status`` — debug helper for notebook environments.
    llm_services:  ``get_chat_llm`` — provider-agnostic chat LLM factory;
        ``get_reasoning_llm`` — DeepSeek-backed reasoning wrapper;
        ``get_strong_llm`` — config-driven high-capability wrapper;
        ``list_available_chat_providers`` — API key availability mapping.
"""

from .embeddings import (
    get_default_embeddings,
    get_small_embeddings,
    get_large_embeddings,
    print_provider_status,
    list_available_providers,
    detect_available_providers,
)

from .llm_services import (
    get_chat_llm,
    get_reasoning_llm,
    get_strong_llm,
    list_available_chat_providers,
)

__all__ = [
    # Embedding functions
    "get_default_embeddings",
    "get_small_embeddings",
    "get_large_embeddings",
    "print_provider_status",
    "list_available_providers",
    "detect_available_providers",
    # Chat functions
    "get_chat_llm",
    "get_reasoning_llm",
    "get_strong_llm",
    "list_available_chat_providers",
]