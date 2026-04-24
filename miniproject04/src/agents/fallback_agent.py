"""
Fallback Agent Module.

Handles persistent system failures, database execution errors, 
or unparseable LLM outputs by returning a graceful, user-friendly 
natural language response instead of crashing.
"""

from langfuse import observe

class FallbackAgent:
    """Agent responsible for graceful error handling and recovery."""

    def __init__(self):
        """Initializes the Fallback Agent."""
        pass

    @observe(as_type="generation")
    def handle_error(self, error_type: str, error_details: str, original_query: str) -> dict:
        """
        Generates a safe, user-friendly fallback response based on the error.
        
        Args:
            error_type (str): The category of the error (e.g., 'db_execution', 'validation_failed').
            error_details (str): The technical error message.
            original_query (str): The user's original prompt.
            
        Returns:
            dict: A standardized response package.
        """
        print(f"⚠️ Fallback Agent Triggered | Type: {error_type}")
        
        if error_type == "validation_failed":
            message = "I understood your question, but I'm having trouble creating a safe database query for it. Could you try rephrasing?"
        elif error_type == "db_execution":
            message = "I generated the query, but the database rejected it. This usually happens if the data structure is a bit different than expected."
        elif error_type == "routing_error":
            message = "I encountered an internal error trying to route your request. Please try asking again."
        else:
            message = "I encountered an unexpected system error while processing that request."

        return {
            "type": "error",
            "content": message,
            "metadata": {"technical_details": error_details, "query": original_query},
            "agents_invoked": 2 
        }