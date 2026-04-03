"""
Prompts sub-package for the Prime Lands Real Estate Intelligence Platform.

Exposes the LangChain prompt templates and builder functions used across
the RAG, CAG, and CRAG service layers.  All constants are defined at import
time to take advantage of KV-cache reuse across multi-turn conversations.

Modules:
    rag_templates: Static prompt constants (``RAG_TEMPLATE``, ``SYSTEM_HEADER``,
        ``EVIDENCE_SLOT``, ``USER_SLOT``, ``ASSISTANT_GUIDANCE``) and builder
        functions (``build_rag_prompt``, ``build_system_message``).
"""

from .rag_templates import (
    RAG_TEMPLATE,
    SYSTEM_HEADER,
    EVIDENCE_SLOT,
    USER_SLOT,
    ASSISTANT_GUIDANCE,
    build_rag_prompt,
    build_system_message,
)

__all__ = [
    # Prompt templates
    "RAG_TEMPLATE",
    "SYSTEM_HEADER",
    "EVIDENCE_SLOT",
    "USER_SLOT",
    "ASSISTANT_GUIDANCE",
    # Builder functions
    "build_rag_prompt",
    "build_system_message",
]