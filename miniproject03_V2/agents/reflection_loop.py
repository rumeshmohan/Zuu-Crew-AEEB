# agents/reflection_loop.py

import os
import json
from utils.llm_services import get_llm

PROFILES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "profiles.json")

# ── Food keywords — only these categories need allergen reflection ─────
FOOD_KEYWORDS = [
    "cake", "chocolate", "sweet", "candy", "cheese", "biscuit", "cookie", "pudding",
    "ice cream", "dessert", "food", "snack", "hamper", "fruit", "nuts",
    "cashew", "almond", "peanut", "walnut", "pastry", "bread", "muffin",
    "brownie", "toffee", "fudge", "truffle", "praline", "nougat", "wafer",
]

# ── Prompts ────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are a strict safety reviewer. 
You must strictly follow the output format rules. 
CRITICAL: DO NOT output your internal reasoning, thinking process, or any conversational filler under any circumstances. Output ONLY the final drafted text or the REVISED text.
"""

REFLECTION_PROMPT = """
You are the Kapruka Safety Reflection Agent.
USER QUERY: {query}
ALLERGIES ON FILE: {allergies}
DRAFT RECOMMENDATION: {draft}

TASK: Check if the DRAFT RECOMMENDATION violates any of the ALLERGIES ON FILE.
- If SAFE: Output ONLY the original DRAFT RECOMMENDATION exactly as written.
- If UNSAFE: Output exactly "REVISED: " followed by a polite apology that explains *specifically why* the item was blocked (e.g., mentioning the specific ingredient that triggered the allergy) and ask them if they'd like a safe alternative.
"""


def _is_food_related(query: str, draft: str) -> bool:
    """Return True only if the query or draft is about food/edible products."""
    combined = (query + " " + draft).lower()
    return any(keyword in combined for keyword in FOOD_KEYWORDS)


def load_recipient_profile(customer_id: str, recipient: str) -> dict:
    """Load a recipient's profile from profiles.json."""
    with open(PROFILES_PATH, "r") as f:
        return json.load(f).get(customer_id, {}).get("recipients", {}).get(recipient, {})


def run_reflection(query: str, draft: str, customer_id: str, recipient: str) -> str:
    """
    Final safety gate for product recommendations.
    Only runs allergen check for food-related products.
    Accepts and returns plain text only — image_urls are handled
    separately in app.py and are never passed through this gate.
    """
    # ── Skip reflection entirely for non-food products ────────────────
    if not _is_food_related(query, draft):
        return draft

    try:
        profile = load_recipient_profile(customer_id, recipient)
    except FileNotFoundError:
        return draft

    allergies = profile.get("allergies", [])
    if not allergies:
        return draft

    prompt = REFLECTION_PROMPT.format(query=query, allergies=allergies, draft=draft)

    try:
        return get_llm(tier="strong").generate(prompt=prompt, system_prompt=SYSTEM_PROMPT).strip()
    except Exception as e:
        print(f"⚠️ Reflection error: {e}")
        return draft