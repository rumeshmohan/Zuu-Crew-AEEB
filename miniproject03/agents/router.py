import os
import re
from openai import OpenAI
from utils.config import get_config, get_api_key

config = get_config()
PROVIDER = config.get("provider.default", "groq")
MODEL = config.get_model(PROVIDER, "router")

try:
    api_key = "ollama-local" if PROVIDER == "ollama" else get_api_key(PROVIDER)
except ValueError as e:
    raise ValueError(f"Router agent error: {e}")

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

ROUTER_PROMPT = """
Analyze the user's query and categorize it into exactly one of these intents:
[CATALOG]    - Searching for products to buy right now (cakes, flowers, gifts).
[LOGISTICS]  - Questions about delivery areas (Kandy, Colombo), times, or shipping.
[PREFERENCE] - User stating facts to be saved about someone's likes, loves, dislikes, or allergies.
[CHITCHAT]   - Greetings, thanks, or general talk.

Output ONLY the intent tag.
"""

CHITCHAT_KEYWORDS  = {"hi", "hello", "hey", "thanks", "bye", "ayubowan"}
LOGISTICS_KEYWORDS = {"delivery", "shipping", "kandy", "colombo", "cost", "fee"}
PREFERENCE_KEYWORDS = {"allergic", "allergy", "prefer", "likes", "dislikes", "favorite", "love", "loves", "hates"}


def route_query(query: str) -> str:
    """Determine intent using keyword matching, with LLM fallback."""
    clean_query = re.sub(r'[^\w\s]', '', query.lower().strip())
    words = set(clean_query.split())

    if words & CHITCHAT_KEYWORDS or "how are you" in clean_query:
        return "[CHITCHAT]"

    if words & LOGISTICS_KEYWORDS or "how much" in clean_query:
        return "[LOGISTICS]"

    if words & PREFERENCE_KEYWORDS:
        return "[PREFERENCE]"

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": ROUTER_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return "[CATALOG]"