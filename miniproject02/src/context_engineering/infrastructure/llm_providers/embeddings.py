"""
Embedding provider factory for the Prime Lands Real Estate Intelligence Platform.

Provides a unified ``get_default_embeddings`` factory that instantiates a
LangChain embedding model for whichever provider is configured in
``config.yaml``, along with convenience wrappers for common model sizes and
provider introspection utilities.

Supported providers:
    openai:      OpenAI text-embedding-3-small / text-embedding-3-large.
    huggingface: Local sentence-transformers models (no API key required).
    cohere:      Cohere embed-english-v3.0 and multilingual models
                 (``langchain-cohere``).
"""

from typing import Optional, Any, List
import os
from langchain_openai import OpenAIEmbeddings

from context_engineering.config import (
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    get_embedding_api_key,
)


def get_default_embeddings(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs: Any
) -> OpenAIEmbeddings:
    """Create and return a LangChain embedding model for the configured provider.

    Reads provider and model defaults from ``config.yaml`` and accepts
    per-call overrides.  Provider-specific client classes and API keys are
    resolved internally so callers remain provider-agnostic.

    Args:
        model (Optional[str]): Embedding model identifier.  Defaults to
            ``EMBEDDING_MODEL`` from config if not supplied.
        provider (Optional[str]): Provider name override.  Must be one of
            ``openai``, ``huggingface``, or ``cohere``.  Defaults to
            ``EMBEDDING_PROVIDER`` from config.
        **kwargs (Any): Additional keyword arguments forwarded directly to the
            underlying provider client constructor.

    Returns:
        OpenAIEmbeddings: A LangChain embeddings instance ready for use.
            The concrete type varies by provider (e.g. ``HuggingFaceEmbeddings``,
            ``CohereEmbeddings``) but all share the ``OpenAIEmbeddings`` interface.

    Raises:
        ImportError: If the required provider package is not installed.
        ValueError: If a required API key environment variable is missing,
            or if an unrecognised provider name is supplied.

    Examples:
        >>> embeddings = get_default_embeddings()
        >>> embeddings = get_default_embeddings(model="text-embedding-3-large")
        >>> embeddings = get_default_embeddings(provider="openai")
    """
    use_provider = provider or EMBEDDING_PROVIDER
    use_model = model or EMBEDDING_MODEL

    print(f"Using EMBEDDING provider: {use_provider.upper()}")
    print(f"Model: {use_model}")

    if use_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        return OpenAIEmbeddings(
            model=use_model,
            openai_api_key=api_key,
            **kwargs
        )

    elif use_provider == "huggingface":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            raise ImportError(
                "HuggingFace requires langchain-huggingface. "
                "Install with: pip install langchain-huggingface"
            )

        return HuggingFaceEmbeddings(
            model_name=use_model,
            **kwargs
        )

    elif use_provider == "cohere":
        try:
            from langchain_cohere import CohereEmbeddings
        except ImportError:
            raise ImportError(
                "Cohere requires langchain-cohere. "
                "Install with: uv add langchain-cohere"
            )

        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in environment")

        return CohereEmbeddings(
            model=use_model,
            cohere_api_key=api_key,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown embedding provider '{use_provider}'. "
            f"Available: openai, huggingface, cohere"
        )


def get_small_embeddings(**kwargs: Any) -> OpenAIEmbeddings:
    """Return a small, fast OpenAI embedding model suited for high-volume processing.

    Uses ``text-embedding-3-small`` (1536 dimensions).  Recommended for
    prototyping and scenarios where throughput matters more than accuracy.

    Args:
        **kwargs (Any): Additional arguments forwarded to ``get_default_embeddings``.

    Returns:
        OpenAIEmbeddings: OpenAI embeddings instance using ``text-embedding-3-small``.
    """
    return get_default_embeddings(
        model="text-embedding-3-small",
        provider="openai",
        **kwargs
    )


def get_large_embeddings(**kwargs: Any) -> OpenAIEmbeddings:
    """Return a large, high-quality OpenAI embedding model suited for accuracy-critical tasks.

    Uses ``text-embedding-3-large`` (3072 dimensions).  Recommended when
    retrieval precision matters more than throughput.

    Args:
        **kwargs (Any): Additional arguments forwarded to ``get_default_embeddings``.

    Returns:
        OpenAIEmbeddings: OpenAI embeddings instance using ``text-embedding-3-large``.
    """
    return get_default_embeddings(
        model="text-embedding-3-large",
        provider="openai",
        **kwargs
    )


def detect_available_providers() -> List[str]:
    """Detect which embedding providers are currently available in the environment.

    Availability is determined by both package installation and, where required,
    the presence of the corresponding API key environment variable.
    HuggingFace is always considered available if its package is installed
    since it runs locally without an API key.

    Returns:
        List[str]: Provider name strings for all ready-to-use providers,
        in detection order: ``openai``, ``huggingface``, ``cohere``.
    """
    available = []

    try:
        import langchain_openai
        if os.getenv("OPENAI_API_KEY"):
            available.append("openai")
    except ImportError:
        pass

    try:
        import langchain_huggingface
        available.append("huggingface")
    except ImportError:
        pass

    try:
        import langchain_cohere
        if os.getenv("COHERE_API_KEY"):
            available.append("cohere")
    except ImportError:
        pass

    return available


def list_available_providers() -> List[str]:
    """Return a list of available embedding provider names.

    Delegates to ``detect_available_providers`` for runtime detection.

    Returns:
        List[str]: Provider name strings for all currently available providers.
    """
    return detect_available_providers()


def print_provider_status() -> None:
    """Print a formatted status report of all embedding providers to stdout.

    Checks each provider's package installation and API key presence, then
    prints availability, supported models, and a recommendation if the
    configured provider is unavailable.  Useful for debugging configuration
    issues in notebook environments.
    """
    providers = detect_available_providers()

    print("\n" + "="*60)
    print("EMBEDDING PROVIDER STATUS")
    print("="*60)

    has_openai_key = bool(os.getenv("OPENAI_API_KEY"))
    has_cohere_key = bool(os.getenv("COHERE_API_KEY"))

    print("\nEmbedding Providers:")
    print(f"   OpenAI: {'Available' if 'openai' in providers else 'Not configured'}")

    if has_openai_key:
        print(f"      - text-embedding-3-small (1536 dim, fast)")
        print(f"      - text-embedding-3-large (3072 dim, quality)")
    else:
        print(f"      OPENAI_API_KEY not set")

    print(f"   HuggingFace: {'Available' if 'huggingface' in providers else 'Package not installed'}")
    if 'huggingface' in providers:
        print(f"      - Local embeddings (no API key needed)")
        print(f"      - sentence-transformers models")

    print(f"   Cohere: {'Available' if 'cohere' in providers else 'Not configured'}")
    if has_cohere_key:
        print(f"      - embed-english-v3.0 (1024 dim)")
        print(f"      - embed-multilingual-v3.0 (1024 dim)")
    else:
        print(f"      COHERE_API_KEY not set")

    print(f"\nCurrent Config:")
    print(f"   Provider: {EMBEDDING_PROVIDER.upper()}")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   Status: {'Available' if EMBEDDING_PROVIDER in providers else 'Not available'}")

    if not providers:
        print(f"\nNo embedding providers configured!")
        print(f"Install langchain-huggingface for free local embeddings")
        print(f"   OR add OPENAI_API_KEY / COHERE_API_KEY to your .env file")
    elif EMBEDDING_PROVIDER not in providers:
        print(f"\nWarning: Configured provider '{EMBEDDING_PROVIDER}' not available")
        print(f"Available providers: {', '.join(providers)}")

    print("="*60 + "\n")


if __name__ == "__main__":
    print_provider_status()

    try:
        embeddings = get_default_embeddings()
        print("\nSuccessfully created embeddings instance")

        test_vector = embeddings.embed_query("Hello world")
        print(f"Embedding dimension: {len(test_vector)}")

    except Exception as e:
        print(f"\nError: {e}")