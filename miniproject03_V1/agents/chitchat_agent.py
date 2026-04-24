import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from utils.llm_services import get_llm

SYSTEM_PROMPT = """
You are a friendly, welcoming Kapruka AI Assistant.
Your goal is to greet the user, acknowledge their message, and politely guide them to ask about shopping, gifts, or our products.

RULES:
1. Keep it brief (1-2 sentences max).
2. Be warm and distinctly Sri Lankan in your hospitality (e.g., using "Ayubowan" occasionally, but don't overdo it).
3. End by asking how you can help them shop today.
4. Do NOT attempt to answer product or delivery questions. If they ask about those, just say you are the greeter and will connect them with the right specialist.
5. Do not explicitly mention the user's saved allergies or preferences in your greetings.
"""


def handle_chitchat_query(query: str) -> str:
    """Handle basic greetings, thank yous, and off-topic chatter."""
    try:
        return get_llm(tier="general").generate(prompt=query, system_prompt=SYSTEM_PROMPT).strip()
    except Exception as e:
        print(f"⚠️ Chitchat error: {e}")
        return "Ayubowan! Welcome to Kapruka. How can I help you find the perfect gift today?"