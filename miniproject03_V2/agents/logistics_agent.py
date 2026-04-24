import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.llm_services import get_llm

POLICY_PATH = os.path.join(root_dir, "data", "logistics_policy.txt")
FALLBACK_POLICY = "Delivery takes 1-5 days. Standard fee is Rs. 400."


def load_logistics_knowledge() -> str:
    """Read the Kapruka logistics policies from the external text file."""
    try:
        with open(POLICY_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"⚠️ Warning: Could not find {POLICY_PATH}. Using fallback knowledge.")
        return FALLBACK_POLICY


KNOWLEDGE_BASE = load_logistics_knowledge()

SYSTEM_PROMPT = """
You are the Kapruka Customer Service Agent handling Logistics and Policies.

RULES:
1. Be polite, clear, and concise (under 3 sentences).
2. Answer ONLY from the KNOWLEDGE BASE provided in the prompt.
3. If the user asks something not covered in the Knowledge Base, apologize and say you don't have that information.
4. Format prices nicely (e.g., Rs. 400).
"""


def handle_logistics_query(query: str) -> str:
    """Handle questions about shipping, returns, and delivery."""
    try:
        prompt = (
            f"KNOWLEDGE BASE:\n{KNOWLEDGE_BASE}\n\n"
            f"User Query: {query}"
        )
        return get_llm(tier="general").generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT
        )
    except Exception as e:
        return f"I'm sorry, our logistics system is currently unavailable. Error: {e}"