"""
SQL Generator Agent Module.

Takes a natural language query, injects the dynamic schema, and uses the configured LLM
to generate a PostgreSQL query. Implements a 3-attempt retry loop.
"""

from src.engine.prompt_builder import build_sql_generator_prompt
from src.engine.sql_validator import SQLValidator
from src.utils.llm_services import get_llm
from src.utils.config import get_config
from langfuse import observe

class SQLGeneratorAgent:
    """Agent responsible for translating natural language to valid SQL."""

    def __init__(self, tier: str = None):
        """Initializes the LLM (using 'strong' tier for coding) and the SQL Validator."""
        config = get_config()
        selected_tier = tier or config.get("provider.tier", "strong") 
        
        self.llm = get_llm(tier=selected_tier) 
        self.validator = SQLValidator()
        self.max_retries = 3

    def clean_sql(self, raw_text: str) -> str:
        """Strips markdown formatting from the LLM output."""
        return raw_text.replace("```sql", "").replace("```", "").strip()

    @observe(as_type="generation")
    def generate_sql(self, user_query: str, chat_history: list = None) -> dict:
        """Generates SQL from a user query, validating and retrying up to 3 times."""
        system_prompt = build_sql_generator_prompt()
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Inject the last 4 chat messages for context
        if chat_history:
            for msg in chat_history[-4:]:
                if msg["role"] in ["user", "assistant"]:
                    messages.append({"role": msg["role"], "content": str(msg["content"])})
                    
        messages.append({"role": "user", "content": f"User Request: {user_query}"})

        for attempt in range(1, self.max_retries + 1):
            print(f"--- SQL Generation Attempt {attempt}/3 (Model: {self.llm.model}) ---")
            
            raw_response = self.llm.generate_messages(messages)
            generated_sql = self.clean_sql(raw_response)
            
            validation_result = self.validator.validate_query(generated_sql)
            
            if validation_result["is_valid"]:
                return {
                    "status": "success",
                    "sql": generated_sql,
                    "message": "Query generated successfully."
                }
            
            print(f"Validation Failed: {validation_result['message']}")
            messages.append({"role": "assistant", "content": generated_sql})
            messages.append({
                "role": "user", 
                "content": f"Error: {validation_result['message']}. Please fix the query and return ONLY the valid SQL."
            })

        return {
            "status": "error",
            "sql": None,
            "message": "I'm having trouble translating that into a safe database query. Could you please rephrase or clarify your request?"
        }