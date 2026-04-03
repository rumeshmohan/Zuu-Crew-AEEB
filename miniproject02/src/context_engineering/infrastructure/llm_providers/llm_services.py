"""
LLM provider factory for the Prime Lands Real Estate Intelligence Platform.

Provides a unified ``get_chat_llm`` factory that instantiates a LangChain
chat model for whichever provider is configured in ``config.yaml``, along
with convenience wrappers for specialised use cases.

Supported providers:
    groq:       Fast free inference via Groq Cloud (``langchain-groq``).
    google:     Google Gemini free tier (``langchain-google-genai``).
    deepseek:   Cost-effective reasoning model via DeepSeek API.
    openrouter: Unified access to many models via OpenRouter.
    openai:     Direct OpenAI API access.
    cohere:     Cohere Command models with native RAG grounding (``langchain-cohere``).
"""

from typing import Optional, Any
import os
from langchain_openai import ChatOpenAI

from context_engineering.config import (
    CHAT_PROVIDER,
    CHAT_MODEL,
    OPENROUTER_BASE_URL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    LLM_STREAMING,
    get_chat_api_key,
)


def get_chat_llm(
    model: Optional[str] = None,
    provider: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    streaming: Optional[bool] = None,
    **kwargs: Any
) -> ChatOpenAI:
    """Create and return a LangChain chat LLM for the configured provider.

    Reads provider and model defaults from ``config.yaml`` and accepts
    per-call overrides for all key parameters.  Provider-specific client
    classes and API keys are resolved internally so callers remain
    provider-agnostic.

    Args:
        model (Optional[str]): Model identifier string.  Defaults to
            ``CHAT_MODEL`` from config if not supplied.
        provider (Optional[str]): Provider name override.  Must be one of
            ``groq``, ``google``, ``gemini``, ``deepseek``, ``openrouter``,
            or ``openai``.  Defaults to ``CHAT_PROVIDER`` from config.
        temperature (Optional[float]): Sampling temperature where 0.0 is
            fully deterministic.  Defaults to ``LLM_TEMPERATURE`` from config.
        max_tokens (Optional[int]): Maximum tokens in the generated response.
            Defaults to ``LLM_MAX_TOKENS`` from config.
        streaming (Optional[bool]): Enable token streaming.  Defaults to
            ``LLM_STREAMING`` from config.
        **kwargs (Any): Additional keyword arguments forwarded directly to the
            underlying provider client constructor.

    Returns:
        ChatOpenAI: A LangChain chat model instance ready for invocation.
            The concrete type varies by provider (e.g. ``ChatGroq``,
            ``ChatGoogleGenerativeAI``, ``ChatCohere``) but all share the
            ``ChatOpenAI`` interface.

    Raises:
        ImportError: If the required provider package is not installed.
        ValueError: If a required API key environment variable is missing,
            or if an unrecognised provider name is supplied.

    Examples:
        >>> llm = get_chat_llm()
        >>> llm = get_chat_llm(model="llama-3.1-70b-versatile", provider="groq")
        >>> llm = get_chat_llm(provider="gemini")
    """
    use_provider = provider or CHAT_PROVIDER
    use_model = model or CHAT_MODEL

    use_temperature = temperature if temperature is not None else LLM_TEMPERATURE
    use_max_tokens = max_tokens if max_tokens is not None else LLM_MAX_TOKENS
    use_streaming = streaming if streaming is not None else LLM_STREAMING

    print(f"Using CHAT provider: {use_provider.upper()}")
    print(f"Model: {use_model}")

    if use_provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError("Groq requires langchain-groq. Install with: uv add langchain-groq")

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")

        return ChatGroq(
            model=use_model,
            groq_api_key=api_key,
            temperature=use_temperature,
            max_tokens=use_max_tokens,
            **kwargs
        )

    elif use_provider in ["google", "gemini"]:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Google Gemini requires langchain-google-genai. Install with: uv add langchain-google-genai")

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment")

        return ChatGoogleGenerativeAI(
            model=use_model,
            google_api_key=api_key,
            temperature=use_temperature,
            max_output_tokens=use_max_tokens,
            **kwargs
        )

    elif use_provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment")

        return ChatOpenAI(
            model=use_model,
            temperature=use_temperature,
            max_tokens=use_max_tokens,
            streaming=use_streaming,
            openai_api_key=api_key,
            openai_api_base="https://api.deepseek.com/v1",
            **kwargs
        )

    elif use_provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found in environment")

        return ChatOpenAI(
            model=use_model,
            temperature=use_temperature,
            max_tokens=use_max_tokens,
            streaming=use_streaming,
            openai_api_key=api_key,
            openai_api_base=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/zuu-crew/context-engineering",
                "X-Title": "Context Engineering RAG"
            },
            **kwargs
        )

    elif use_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")

        return ChatOpenAI(
            model=use_model,
            temperature=use_temperature,
            max_tokens=use_max_tokens,
            streaming=use_streaming,
            openai_api_key=api_key,
            **kwargs
        )

    elif use_provider == "cohere":
        try:
            from langchain_cohere import ChatCohere
        except ImportError:
            raise ImportError("Cohere requires langchain-cohere. Install with: uv add langchain-cohere")

        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found in environment")

        return ChatCohere(
            model=use_model,
            cohere_api_key=api_key,
            temperature=use_temperature,
            max_tokens=use_max_tokens,
            **kwargs
        )

    else:
        raise ValueError(
            f"Unknown provider '{use_provider}'. "
            f"Available: groq, google, gemini, deepseek, openrouter, openai, cohere"
        )


def get_reasoning_llm(**kwargs: Any):
    """Return a reasoning-optimised LLM using the DeepSeek provider.

    DeepSeek is selected for reasoning tasks due to its strong chain-of-thought
    performance at low cost.

    Args:
        **kwargs (Any): Additional arguments forwarded to ``get_chat_llm``.

    Returns:
        ChatOpenAI: DeepSeek-backed chat model instance.
    """
    return get_chat_llm(provider="deepseek", **kwargs)


def get_strong_llm(**kwargs: Any):
    """Return a high-capability LLM using the default configured provider.

    Delegates to ``get_chat_llm`` with no provider override, so the model
    and provider are driven entirely by ``config.yaml``.

    Args:
        **kwargs (Any): Additional arguments forwarded to ``get_chat_llm``.

    Returns:
        ChatOpenAI: Chat model instance for the configured best-available model.
    """
    return get_chat_llm(**kwargs)


def list_available_chat_providers() -> dict:
    """Return a mapping of provider names to their API key availability.

    Checks for the presence of each provider's expected environment variable(s)
    and returns ``True`` for providers that are ready to use.

    Returns:
        dict: Keys are provider name strings; values are ``bool`` indicating
        whether the corresponding API key environment variable is set.
    """
    return {
        'groq': bool(os.getenv("GROQ_API_KEY")),
        'google': bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
        'deepseek': bool(os.getenv("DEEPSEEK_API_KEY")),
        'openrouter': bool(os.getenv("OPENROUTER_API_KEY")),
        'openai': bool(os.getenv("OPENAI_API_KEY")),
        'cohere': bool(os.getenv("COHERE_API_KEY")),
    }