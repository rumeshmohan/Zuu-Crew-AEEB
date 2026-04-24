# agents/catalog_agent.py

import os
import sys
import re

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from openai import OpenAI
from qdrant_client.http import models as rest
from utils.config import get_config, get_api_key
from memory.vector_db import retrieve_products

config   = get_config()
PROVIDER = config.get("provider.default", "groq")
MODEL    = config.get_model(PROVIDER, "general")

try:
    api_key = "ollama-local" if PROVIDER == "ollama" else get_api_key(PROVIDER)
except ValueError as e:
    raise ValueError(f"Catalog agent error: {e}")

BASE_URL_MAP = {
    "ollama":      "http://localhost:11434/v1",
    "openrouter":  "https://openrouter.ai/api/v1",
    "groq":        "https://api.groq.com/openai/v1",
    "gemini":      "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openai":      "https://api.openai.com/v1",
    "cohere":      "https://api.cohere.ai/compatibility/v1",
    "deepseek":    "https://api.deepseek.com/v1",
    "anthropic":   "https://api.anthropic.com/v1",
}

client = OpenAI(api_key=api_key, base_url=BASE_URL_MAP.get(PROVIDER))

# ── Prompts ────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a helpful Kapruka sales assistant.
Recommend products strictly from the 'Database Context'.
Rules:
1. Mention exact product name and price (Rs. XXXX).
2. If context is empty, apologize politely.
3. Stay under 4 sentences.
4. ONLY mention allergy or safety information if the products are food or edible items.
   For flowers, gifts, electronics, or non-food items, do NOT mention allergies at all.
"""

KNOWN_TAGS     = ["eggless", "vegetarian", "sugar-free", "alcohol-free"]
BUDGET_PATTERN = re.compile(r"Budget:\s*LKR\s*(\d+)[-–](\d+)")


# ── Filter helpers ─────────────────────────────────────────

def extract_filters(query: str) -> dict | None:
    """Extract metadata keywords and budget range from the query for hard filtering."""
    filters = {}

    active_tags = [tag for tag in KNOWN_TAGS if tag in query.lower()]
    if active_tags:
        filters["tags"] = active_tags

    budget_match = BUDGET_PATTERN.search(query)
    if budget_match:
        filters["min_price"] = int(budget_match.group(1))
        filters["max_price"] = int(budget_match.group(2))

    return filters or None


def build_qdrant_filter(metadata_filters: dict) -> rest.Filter | None:
    """Build a Qdrant filter object from extracted metadata filters."""
    must_conditions = []

    if "tags" in metadata_filters:
        must_conditions.append(
            rest.FieldCondition(key="tags", match=rest.MatchAny(any=metadata_filters["tags"]))
        )

    if "min_price" in metadata_filters and "max_price" in metadata_filters:
        must_conditions.append(
            rest.FieldCondition(
                key="price",
                range=rest.Range(
                    gte=metadata_filters["min_price"],
                    lte=metadata_filters["max_price"],
                ),
            )
        )

    return rest.Filter(must=must_conditions) if must_conditions else None


# ── Main handler ───────────────────────────────────────────────────────

def handle_catalog_query(query: str, customer_id: str = "default", history: str = "") -> dict:
    """
    Retrieve matching products from Qdrant (retrieval mode is determined by
    embedding.provider and embedding.clip_mode in params.yaml), generate an
    LLM response, and return BOTH the text response and any product image URLs.

    Retrieval modes (configured in params.yaml — no code change needed):
      embedding.provider = clip, clip_mode = text_image  → dual text + image vector search
      embedding.provider = clip, clip_mode = text_only   → text vector search only
      embedding.provider = cohere / openai / …           → API embedding, text search only

    Returns:
        {
            "text":       str,         # LLM-generated recommendation text
            "image_urls": list[str],   # product image URLs for UI rendering
        }
    """
    # ── Retrieve products via configured embedding backend ─────────────
    metadata_filters = extract_filters(query)
    q_filter         = build_qdrant_filter(metadata_filters) if metadata_filters else None
    raw_results      = retrieve_products(query, limit=5, search_filter=q_filter)

    # ── Build LLM context ──────────────────────────────────────────────
    context = (
        "\n".join(
            f"- {p.get('name')} | Price: Rs.{p.get('price')} | URL: {p.get('product_url', '')}"
            for p in raw_results
        )
        or "No matches."
    )

    # ── Collect image URLs for UI rendering ───────────────────────────
    # Image URLs come from the product payload (always stored regardless of
    # embedding mode), so the image grid works in all three retrieval modes.
    image_urls = [
        p["image_url"]
        for p in raw_results
        if p.get("image_url")
    ]

    user_msg = f"History:\n{history}\n\nQuery: {query}\n\nContext:\n{context}"

    # ── LLM call ──────────────────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,
        )
        text = response.choices[0].message.content.strip()

    except Exception as e:
        text = f"Catalog error: {e}"

    return {
        "text":       text,
        "image_urls": image_urls,
    }