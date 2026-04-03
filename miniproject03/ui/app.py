import os
import sys
import time
import datetime
from pathlib import Path

import streamlit as st

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
    if Path(__file__).parent.name == "ui"
    else Path(__file__).resolve().parent
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from memory.session_buffer import SessionBuffer
from agents.router import route_query
from agents.catalog_agent import handle_catalog_query
from agents.logistics_agent import handle_logistics_query
from agents.chitchat_agent import handle_chitchat_query
from agents.reflection_loop import run_reflection
from agents.preference_agent import handle_preference_query

st.set_page_config(
    page_title="Kapruka Concierge",
    page_icon="🛍️",
    layout="centered",
    initial_sidebar_state="expanded",
)

ALLERGENS = ["peanut", "cashew", "nut", "almond", "walnut", "pistachio", "hazelnut", "pecan"]
OCCASIONS = ["🎂 Birthday", "💍 Anniversary", "🌹 Valentine's", "🎉 Just Because", "🎓 Graduation"]
BUDGETS = {
    "Under LKR 2,000": (0, 2000),
    "LKR 2,000–5,000": (2000, 5000),
    "LKR 5,000+":      (5000, 999999),
}

QUICK_REPLY_MAP = {
    "[CATALOG]":    ["Show more options", "Filter by price", "Tell me about delivery"],
    "[LOGISTICS]":  ["Change delivery address", "What are delivery times?"],
    "[CHITCHAT]":   ["Help me pick a gift", "What can you do?", "Show me chocolates"],
    "[PREFERENCE]": ["Browse chocolates", "What do you recommend?"],
}


def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M")


def _has_allergen(text: str) -> bool:
    """Return True if the text mentions an allergen outside of a safe/free-from context."""
    lower = text.lower()
    safe_phrases = [
        "safety from", "free from", "without", "nut-free",
        "no peanuts", "no nuts", "allergy-friendly",
        "allergy", "allergic", "allergies", "do not contain", "does not contain"
    ]
    if any(phrase in lower for phrase in safe_phrases):
        return False
    return any(allergen in lower for allergen in ALLERGENS)


def _build_query_context(user_input: str) -> str:
    """Enrich the user query with sidebar-selected occasion, budget, and allergy needs."""
    extra = []
    if st.session_state.occasion:
        extra.append(f"Occasion: {st.session_state.occasion}.")
    if st.session_state.budget:
        lo, hi = BUDGETS[st.session_state.budget]
        extra.append(f"Budget: LKR {lo}-{hi}.")
    allergies = st.session_state.profile.get("allergies", [])
    if allergies:
        extra.append(f"Must be safe from: {', '.join(allergies)}.")
    return (" ".join(extra) + " " + user_input).strip() if extra else user_input


def _quick_replies_for(intent: str) -> list:
    """Return the quick-reply suggestions for the given intent tag."""
    return QUICK_REPLY_MAP.get(intent, [])


def initialize_session():
    """Initialise all session state keys with their default values on first load."""
    if "buffer" not in st.session_state:
        st.session_state.buffer = SessionBuffer(max_pairs=5)
        st.session_state.buffer.set_persistent_context(
            "Recipient: Wife. Allergies: Peanuts, Cashews, and Tree Nuts. Preferences: Dark Chocolate."
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "Ayubowan! 🙏 Welcome to Kapruka. How can I help you find the perfect gift today?",
            "ts": _now(),
        }]

    defaults = {
        "dark_mode":      True,
        "occasion":       None,
        "budget":         None,
        "quick_replies":  [],
        "_pending_input": None,
        "profile": {
            "customer_id": "CUS_001",
            "recipient":   "Wife",
            "allergies":   ["Peanuts", "Cashews", "Tree Nuts"],
            "preference":  "Dark Chocolate",
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session()

dm = st.session_state.dark_mode

if dm:
    BG_COLOR   = "#0b141a"
    SIDEBAR_BG = "#111b21"
    TEXT_COLOR = "#e9edef"
    ASST_BUB   = "#202c33"
    USER_BUB   = "#005c4b"
else:
    BG_COLOR   = "#efeae2"
    SIDEBAR_BG = "#ffffff"
    TEXT_COLOR = "#111b21"
    ASST_BUB   = "#ffffff"
    USER_BUB   = "#dcf8c6"

st.markdown(f"""
<style>
    .stApp, [data-testid="stBottomBlockContainer"] {{ background-color: {BG_COLOR} !important; }}
    [data-testid="stSidebar"] {{ background-color: {SIDEBAR_BG} !important; }}
    h1, h2, h3, p, span, label {{ color: {TEXT_COLOR} !important; }}
    [data-testid="stChatMessage"] {{
        border-radius: 12px !important;
        padding: 10px 15px !important;
        margin-bottom: 10px !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {{ background-color: {ASST_BUB} !important; }}
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {{ background-color: {USER_BUB} !important; }}
    [data-testid="stChatInput"] {{
        border: 1px solid rgba(150, 150, 150, 0.2) !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
    }}
    header[data-testid="stHeader"] {{ background: transparent !important; }}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🛍️ Kapruka")

    if st.button("🌙 Dark Mode" if dm else "☀️ Light Mode", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("---")

    st.markdown("#### 👤 User Profile")
    p = st.session_state.profile
    st.info(
        f"**Customer:** {p['customer_id']}\n\n"
        f"**Recipient:** 💝 {p['recipient']}\n\n"
        f"**Allergies:** ⚠️ {', '.join(p['allergies'])}\n\n"
        f"**Preference:** 🍫 {p['preference']}"
    )

    st.markdown("#### ⚙️ Context")
    selected_occasion = st.selectbox("Occasion", ["— None —"] + OCCASIONS, label_visibility="collapsed")
    st.session_state.occasion = None if selected_occasion == "— None —" else selected_occasion

    selected_budget = st.selectbox("Budget", ["— Any —"] + list(BUDGETS.keys()), label_visibility="collapsed")
    st.session_state.budget = None if selected_budget == "— Any —" else selected_budget

    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.buffer.clear()
        st.session_state.chat_history = [{
            "role": "assistant",
            "content": "Memory cleared. How can I help? 😊",
            "ts": _now(),
        }]
        st.session_state.quick_replies = []
        st.rerun()

st.markdown("## 🛍️ Kapruka AI Concierge")

for msg in st.session_state.chat_history:
    avatar = "🛍️" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and _has_allergen(msg["content"]):
            st.warning("⚠️ **Allergen alert:** This response mentions nut-based products. Verify before purchasing.")

if st.session_state.quick_replies:
    cols = st.columns(len(st.session_state.quick_replies))
    for idx, reply in enumerate(st.session_state.quick_replies):
        with cols[idx]:
            if st.button(f"💬 {reply}", key=f"qr_{idx}_{reply}", use_container_width=True):
                st.session_state._pending_input = reply
                st.session_state.quick_replies = []
                st.rerun()

raw_input = st.chat_input("Type your message here…")
if st.session_state._pending_input:
    raw_input = st.session_state._pending_input
    st.session_state._pending_input = None

if raw_input:
    user_input = raw_input.strip()
    now_ts = _now()

    st.session_state.chat_history.append({"role": "user", "content": user_input, "ts": now_ts})
    st.session_state.buffer.add_message("user", user_input)

    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)

    try:
        enriched_input = _build_query_context(user_input)
        intent = route_query(enriched_input)
        history_context = st.session_state.buffer.get_history_string()

        if intent == "[CATALOG]":
            with st.status("🧠 Consulting Semantic Memory…", expanded=False) as status:
                st.write(f"Checking recipient profile for {st.session_state.profile['customer_id']}…")
                time.sleep(0.4)
                status.update(label="✅ Profile applied — fetching recommendations", state="complete")

            draft = handle_catalog_query(query=enriched_input, history=history_context)
            raw_response = run_reflection(
                query=enriched_input,
                draft=draft,
                customer_id=st.session_state.profile["customer_id"],
                recipient=st.session_state.profile["recipient"],
            )
            response = raw_response.split("REVISED:")[-1].strip() if "REVISED:" in raw_response else raw_response

        elif intent == "[LOGISTICS]":
            with st.spinner("🚚 Checking delivery systems…"):
                response = handle_logistics_query(query=user_input)

        elif intent == "[PREFERENCE]":
            with st.spinner("📝 Updating semantic memory..."):
                response = handle_preference_query(
                    query=user_input,
                    customer_id=st.session_state.profile["customer_id"]
                )
            st.toast("🧠 Preference saved to Semantic Memory!", icon="💾")

        else:
            with st.spinner("Thinking…"):
                response = handle_chitchat_query(query=user_input)

        st.session_state.quick_replies = _quick_replies_for(intent)

    except Exception as e:
        response = f"Apologies, I ran into a small hiccup 🙏 Please try again. *(Error: {type(e).__name__})*"
        st.session_state.quick_replies = []

    resp_ts = _now()
    st.session_state.buffer.add_message("assistant", response)
    st.session_state.chat_history.append({"role": "assistant", "content": response, "ts": resp_ts})

    with st.chat_message("assistant", avatar="🛍️"):
        st.markdown(response)
        if _has_allergen(response):
            st.warning("⚠️ **Allergen alert:** This response mentions nut-based products. Verify before purchasing.")

    st.rerun()