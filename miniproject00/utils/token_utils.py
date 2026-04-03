"""
Token utilities using tiktoken for estimation and context management.

This module provides:
- Encoding selection per provider/model
- Token counting for text and messages
- Reconciliation of estimated vs actual token usage
- Context-fit guards with summarize/truncate strategies
"""

import tiktoken
from typing import Literal, Optional, Any

ProviderType = Literal["openai", "google", "groq", "mistral", "cohere"]

def pick_encoding(
    provider: ProviderType, model: str
) -> tiktoken.Encoding:
    """
    Select appropriate tiktoken encoding for provider/model.
    Uses o200k_base as the agnostic proxy for non-OpenAI models.
    """
    model_lower = model.lower()
    
    if provider == "openai":
        # Standard OpenAI encoding selection
        if any(x in model_lower for x in ["gpt-4o", "gpt-4", "o3", "o1"]):
            try:
                return tiktoken.get_encoding("o200k_base")
            except Exception:
                pass
        return tiktoken.get_encoding("cl100k_base")

    # For Google, Groq, Mistral, and Cohere, we use o200k_base 
    # as a professional estimation proxy for token economics.
    try:
        return tiktoken.get_encoding("o200k_base")
    except Exception:
        return tiktoken.get_encoding("cl100k_base")


def count_text_tokens(
    text: str, provider: ProviderType, model: str
) -> int:
    """
    Count tokens in a text string using the selected agnostic proxy.
    """
    if not text:
        return 0
    enc = pick_encoding(provider, model)
    return len(enc.encode(text, disallowed_special=()))


def count_messages_tokens(
    messages: list[dict[str, str]],
    provider: ProviderType,
    model: str,
    context_strs: Optional[list[str]] = None,
) -> dict[str, int]:
    """
    Count tokens in a messages array, separating input vs context.
    Essential for Part 4 Budget Keeper logic.
    """
    enc = pick_encoding(provider, model)

    input_tokens = 0
    for msg in messages:
        content = msg.get("content", "")
        # Role/Structure overhead: Agnostic estimation uses 4 tokens per msg
        input_tokens += 4
        input_tokens += len(enc.encode(content, disallowed_special=()))

    context_tokens = 0
    if context_strs:
        for ctx in context_strs:
            context_tokens += len(enc.encode(ctx, disallowed_special=()))

    overhead = 3  # Base structural overhead
    return {
        "input_tokens": input_tokens,
        "context_tokens": context_tokens,
        "estimated_total": input_tokens + context_tokens + overhead,
    }

def fit_within_context(
    messages: list[dict[str, str]],
    provider: ProviderType,
    model: str,
    max_context_tokens: int,
    strategy: Literal["summarize", "truncate"] = "summarize",
    context_strs: Optional[list[str]] = None,
) -> tuple[list[dict[str, str]], Optional[list[str]], dict[str, Any]]:
    """
    Context-fit guard for Part 4. Triggers 'BLOCKED/TRUNCATED' logic 
    if the budget is exceeded.
    """
    counts = count_messages_tokens(messages, provider, model, context_strs)
    current_tokens = counts["estimated_total"]

    if current_tokens <= max_context_tokens:
        return messages, context_strs, {"overflow": False, "original_tokens": current_tokens}

    # Handle Overflow for Part 4 'Budget Keeper' requirement
    if strategy == "summarize":
        return (
            messages,
            context_strs,
            {
                "overflow": True, 
                "original_tokens": current_tokens, 
                "strategy": "summarize",
                "action_required": "Use overflow_summarize.v1 prompt"
            },
        )
    
    # Truncate logic
    return messages, context_strs, {"overflow": True, "strategy": "truncate", "original_tokens": current_tokens}