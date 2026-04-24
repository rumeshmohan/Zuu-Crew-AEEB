"""
Result Interpreter Agent Module.

Takes raw SQL execution results and the user's original query, and translates
them into a human-readable summary and chart visualization specifications.
"""

import json
from src.utils.llm_services import get_llm
from src.utils.config import get_config
from src.engine.prompt_builder import build_interpreter_prompt
from langfuse import observe

class ResultInterpreterAgent:
    """Agent responsible for interpreting database results for the user."""

    def __init__(self, tier: str = None):
        """
        Initializes the Interpreter Agent.
        
        Args:
            tier (str): Optional override for the model tier. Defaults to params.yaml.
        """
        config = get_config()
        selected_tier = tier or config.get("provider.tier", "general")
        self.llm = get_llm(tier=selected_tier)

    @observe(as_type="generation")
    def interpret_results(self, user_query: str, sql_results: list) -> dict:
        """
        Translates raw DB results into natural language and visualization specs.
        
        Args:
            user_query (str): The original question asked by the user.
            sql_results (list): The raw dictionary rows returned from SQLAlchemy.
            
        Returns:
            dict: Parsed JSON with 'summary' and 'chart' info.
        """
        # 1. Handle empty results before wasting LLM tokens
        if not sql_results:
            return {
                "summary": "I ran the query, but no data matched your request in the database.",
                "chart": {"type": "none", "x_axis": None, "y_axis": None, "reason": "No data available."}
            }

        # 2. Build the messages
        system_prompt = build_interpreter_prompt()
        
        # We pass both the query and the raw data as a JSON string
        human_content = f"User Question: {user_query}\n\nRaw Database Results:\n{json.dumps(sql_results[:5], default=str)}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": human_content}
        ]

        print(f"--- Interpreting Results (Model: {self.llm.model}) ---")
        
        try:
            # 3. Call the LLM and parse the JSON
            raw_response = self.llm.generate_messages(messages)
            cleaned_response = raw_response.replace("```json", "").replace("```", "").strip()
            
            interpretation = json.loads(cleaned_response)
            
            if "summary" in interpretation and isinstance(interpretation["summary"], str):
                interpretation["summary"] = interpretation["summary"].replace("`", "")
                
            return interpretation

        except json.JSONDecodeError:
            print("⚠️ Interpreter failed to return valid JSON.")
            return {
                "summary": "The data was retrieved successfully, but I had trouble formatting the summary.",
                "chart": {"type": "none", "x_axis": None, "y_axis": None, "reason": "Parsing error"}
            }
        except Exception as e:
            print(f"❌ Interpreter error: {e}")
            return {
                "summary": f"An error occurred while analyzing the data: {str(e)}",
                "chart": {"type": "none", "x_axis": None, "y_axis": None, "reason": "System error"}
            }

# Quick test block
if __name__ == "__main__":
    interpreter = ResultInterpreterAgent()
    
    query = "How many doctors are in each department?"
    mock_data = [
        {"department_name": "Cardiology", "doctor_count": 12},
        {"department_name": "Neurology", "doctor_count": 8},
        {"department_name": "Pediatrics", "doctor_count": 15}
    ]
    
    result = interpreter.interpret_results(query, mock_data)
    print("\n--- Final Interpretation ---")
    print(f"Summary: {result.get('summary')}")
    print(f"Chart Recommendation: {result.get('chart')}")