"""
RAG prompt templates for the Prime Lands Real Estate Intelligence Platform.

Provides static prompt components and builder functions used by RAGService,
CAGService, and CRAGService.  The static string constants (``RAG_TEMPLATE``,
``SYSTEM_HEADER``, and the slot fragments) are defined once at import time to
take advantage of KV-cache reuse across multi-turn conversations.

Constants:
    RAG_TEMPLATE:        Full prompt template for RAG chain invocations.
    SYSTEM_HEADER:       Persistent system message for the chat model.
    EVIDENCE_SLOT:       Formatted block for injecting retrieved property data.
    USER_SLOT:           Formatted block for injecting the customer question.
    ASSISTANT_GUIDANCE:  Expected response structure hint appended to prompts.

Functions:
    build_rag_prompt:   Renders ``RAG_TEMPLATE`` with context and question.
    build_system_message: Returns the static ``SYSTEM_HEADER`` string.
"""

# ---------------------------------------------------------------------------
# RAG prompt template
# ---------------------------------------------------------------------------

RAG_TEMPLATE = """You are an AI Sales Assistant for Primelands (Pvt) Ltd, a premier real estate company in Sri Lanka.

YOUR ROLE:
- Provide accurate details about land plots, prices, and locations.
- Explain payment plans and reservation fees clearly.
- Help users find the best property investment based on their needs.

GROUNDING RULES (CRITICAL):
- Use ONLY the information in the CONTEXT below.
- Cite sources inline as [URL] or [Source] from the context.
- If price or location details are missing, explicitly state "Details not available in current records".
- DO NOT hallucinate prices or land sizes.

RESPONSE FORMAT:
1. **Property Highlights**: 2-3 bullet points (Price, Location, Size).
2. **Answer**: Clear, persuasive answer with inline citations.
3. **Next Step**: Suggest contacting the hotline (+94 77 123 4567) or visiting the office.

CONTEXT:
{context}

QUESTION: {question}

Provide your response following the format above."""


# ---------------------------------------------------------------------------
# System prompt — kept static so the KV cache is reused across turns
# ---------------------------------------------------------------------------

SYSTEM_HEADER = """You are a professional Real Estate Assistant for Primelands Sri Lanka.

**Important Guidelines:**
1. Be polite, professional, and persuasive.
2. Only mention properties listed in the context.
3. If the user asks about a specific location not in the context, say you don't have info on that area yet.
4. Always emphasize "Investment Value" and "Convenience".
5. Keep answers concise (under 150 words unless detailed info is requested).

**Safety Note:** Financial figures (prices/interest rates) must be exact from the context. Do not estimate."""


# ---------------------------------------------------------------------------
# Slot fragments — injected dynamically into multi-turn conversations
# ---------------------------------------------------------------------------

EVIDENCE_SLOT = """
**PROPERTY DATA:**
{evidence}
"""

USER_SLOT = """
**CUSTOMER INQUIRY:**
{question}
"""

ASSISTANT_GUIDANCE = """
**EXPECTED RESPONSE:**
1. Summary: Key property facts (Perch price, Location)
2. Details: Answer the specific question using the text provided
3. Call to Action: Encourage a site visit or phone call
"""


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_rag_prompt(context: str, question: str) -> str:
    """Render the RAG prompt template with retrieved context and a user question.

    Performs a simple ``str.format`` substitution into ``RAG_TEMPLATE``,
    producing a complete, self-contained prompt ready for the LLM.

    Args:
        context (str): Pre-formatted context string produced by ``format_docs``,
            containing source URLs, titles, and content excerpts.
        question (str): The customer's natural-language query.

    Returns:
        str: Fully rendered prompt string combining system instructions,
        grounding rules, context block, and the user question.
    """
    return RAG_TEMPLATE.format(context=context, question=question)


def build_system_message() -> str:
    """Return the static system prompt for the Primelands chat model.

    The system message is defined as a module-level constant so it is
    allocated once and eligible for KV-cache reuse across all turns of a
    conversation.

    Returns:
        str: The ``SYSTEM_HEADER`` string containing role definition,
        behavioural guidelines, and the financial-accuracy safety note.
    """
    return SYSTEM_HEADER