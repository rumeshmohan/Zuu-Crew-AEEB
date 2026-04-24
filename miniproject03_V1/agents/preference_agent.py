import os
import re
import json
from utils.llm_services import get_llm

PROFILES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "profiles.json")

SYSTEM_PROMPT = """
Output ONLY raw JSON. You must use exactly two keys inside the 'updates' object: 'allergies' (list of strings) and 'preferences' (list of strings). 
Do NOT invent new keys like 'likes' or 'dislikes'. 
"""

EXTRACTION_PROMPT = """
Extract recipient preference information ONLY from the current USER MESSAGE.

CRITICAL RULES:
- Do NOT include information from previous parts of the conversation history.
- Only include facts explicitly stated in the message below.
- If the user says they "love", "like", or "want" something, put it in 'preferences'.
- If the user mentions an "allergy" or "cannot eat" something, put it in 'allergies'.

USER MESSAGE: {query}

Return exactly this JSON structure:
{{
    "recipient": "<lowercase name>",
    "updates": {{
        "allergies": [],
        "preferences": []
    }}
}}
"""

CONFIRMATION_PROMPT = """
You are a friendly Kapruka assistant.
A customer just shared preference information about a gift recipient.

RECIPIENT: {recipient}
UPDATES SAVED: {updates}

Confirm warmly in 1-2 sentences that you have noted these preferences and will use them for future recommendations.
"""


def load_profiles() -> dict:
    """Load profiles from profiles.json, returning an empty dict if missing."""
    try:
        with open(PROFILES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_profiles(profiles: dict) -> None:
    """Save profiles back to profiles.json."""
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)


def extract_preference_data(query: str) -> dict:
    """Use LLM to extract structured preference data from the user's message."""
    prompt = EXTRACTION_PROMPT.format(query=query)
    raw = get_llm(tier="general").generate(prompt=prompt, system_prompt=SYSTEM_PROMPT).strip()
    
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw = match.group(0)
    else:
        raw = raw.replace("```json", "").replace("```", "").strip()
        
    return json.loads(raw)


def update_profile(customer_id: str, recipient: str, updates: dict) -> None:
    """Merge extracted updates into the recipient's profile."""
    profiles = load_profiles()
    profile = profiles.setdefault(customer_id, {"recipients": {}})["recipients"].setdefault(recipient, {})

    for key, values in updates.items():
        existing = profile.setdefault(key, [])
        for item in values:
            if item not in existing:
                existing.append(item)

    save_profiles(profiles)


def handle_preference_query(query: str, customer_id: str) -> str:
    """Extract preferences from query, update profiles.json, and confirm to user."""
    try:
        extracted = extract_preference_data(query)
        recipient = extracted.get("recipient", "default").lower()
        updates = extracted.get("updates", {})

        if not updates:
            return "I didn't catch any specific preferences. Could you share more details?"

        update_profile(customer_id, recipient, updates)

        confirmation_prompt = CONFIRMATION_PROMPT.format(recipient=recipient, updates=updates)
        return get_llm(tier="general").generate(
            prompt=confirmation_prompt,
            system_prompt="You are a warm and helpful Kapruka assistant.",
        ).strip()

    except json.JSONDecodeError:
        return "I had trouble understanding those preferences. Could you rephrase?"
    except Exception as e:
        print(f"⚠️ Preference agent error: {e}")
        return "I wasn't able to save those preferences right now. Please try again."