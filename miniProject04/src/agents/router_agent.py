"""
Intent Router Agent Module.

Analyzes incoming user messages and classifies their intent to route them
to the correct downstream agent (e.g., SQL generation vs. general conversation).
"""

import json
from src.utils.llm_services import get_llm
from src.utils.config import get_config
from src.engine.prompt_builder import build_router_prompt
from langfuse import observe

class IntentRouterAgent:
    """Agent responsible for classifying the intent of a user's prompt."""

    def __init__(self, tier: str = None):
        """
        Initializes the Intent Router.
        
        Args:
            tier (str): Optional override for the model tier. If None, it dynamically
                        pulls the default tier from params.yaml.
        """
        config = get_config()
        
        selected_tier = tier or config.get("provider.tier", "general")
        
        self.llm = get_llm(tier=selected_tier)
        
    @observe(as_type="generation")
    def route_query(self, user_query: str, chat_history: list = None) -> dict:
        """Analyzes the query and returns a routing decision."""
        system_prompt = build_router_prompt()
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Inject the last 4 chat messages for context
        if chat_history:
            for msg in chat_history[-4:]:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": str(msg["content"])})
                    
        messages.append({"role": "user", "content": f"User Message: {user_query}"})

        print(f"--- Routing Intent for: '{user_query}' (Model: {self.llm.model}) ---")
        
        try:
            raw_response = self.llm.generate_messages(messages)
            cleaned_response = raw_response.replace("```json", "").replace("```", "").strip()
            
            routing_decision = json.loads(cleaned_response)
            return routing_decision

        except json.JSONDecodeError:
            print("⚠️ Router failed to return valid JSON. Defaulting to sql_generation.")
            return {
                "intent": "sql_generation",
                "reason": "Fallback intent due to parsing error."
            }
        except Exception as e:
            print(f"❌ Routing error: {e}")
            return {
                "intent": "general_chat",
                "reason": "System error, defaulting to safe chat."
            }

# Quick test block
if __name__ == "__main__":
    router = IntentRouterAgent() 
    
    test_queries = [
        "How many doctors are currently active in the Cardiology department?",
        "Hi there, how are you doing today?"
    ]
    
    for q in test_queries:
        result = router.route_query(q)
        print(f"Decision: {result['intent']} | Reason: {result['reason']}\n")