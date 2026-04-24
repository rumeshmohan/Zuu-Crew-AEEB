"""
Main Orchestrator Module.

Ties together the Intent Router, SQL Generator, Database Client, 
Result Interpreter, and Fallback Agent into a single, cohesive pipeline.
"""

from src.agents.router_agent import IntentRouterAgent
from src.agents.sql_agent import SQLGeneratorAgent
from src.agents.interpreter_agent import ResultInterpreterAgent
from src.agents.fallback_agent import FallbackAgent
from src.engine.db_client import DatabaseClient
from langfuse import observe

class NL2SQLPipeline:
    """The main pipeline controlling the flow of data between agents."""

    def __init__(self):
        print("Initializing Agents and Database Client...")
        self.router = IntentRouterAgent()
        self.sql_generator = SQLGeneratorAgent()
        self.interpreter = ResultInterpreterAgent()
        self.fallback = FallbackAgent() 
        self.db_client = DatabaseClient()
        print("Initialization Complete.\n")

    @observe()
    def process_query(self, user_query: str, chat_history: list = None) -> dict:
        """Runs the full multi-agent pipeline for a given user query."""
        print(f"\n{'='*50}\nProcessing Query: '{user_query}'\n{'='*50}")
        
        try:
            # Step 1: Route the Intent
            routing_decision = self.router.route_query(user_query, chat_history)
            
            if routing_decision["intent"] == "general_chat":
                print("-> Routed to General Chat.")
                return {
                    "type": "chat",
                    "content": "I am the Nawaloka Hospital Database Assistant. I can help you query patient records, doctor schedules, billing, and more. How can I assist you with the database today?",
                    "metadata": routing_decision
                }

            # Step 2: Generate SQL
            print("-> Routed to SQL Generator.")
            sql_response = self.sql_generator.generate_sql(user_query, chat_history)
            
            if sql_response["status"] == "error":
                print("-> SQL Generation Failed. Triggering Fallback.")
                # Trigger Fallback Agent
                return self.fallback.handle_error("validation_failed", sql_response["message"], user_query)
                
            generated_sql = sql_response["sql"]
            print(f"-> Generated Valid SQL:\n{generated_sql}")

            # Step 3: Execute SQL on Supabase
            print("-> Executing Query on Database...")
            db_results = self.db_client.execute_query(generated_sql)
            
            # Check for DB execution errors
            if db_results and "error" in db_results[0]:
                print(f"-> Database Execution Error: {db_results[0]['error']}")
                # Trigger Fallback Agent
                return self.fallback.handle_error("db_execution", db_results[0]['error'], user_query)
                
            print(f"-> Database returned {len(db_results)} rows.")

            # Step 4: Interpret Results
            print("-> Interpreting Results...")
            interpretation = self.interpreter.interpret_results(user_query, db_results)

            # Step 5: Return Final Package
            return {
                "type": "data",
                "content": interpretation["summary"],
                "sql": generated_sql,
                "raw_data": db_results,
                "chart_config": interpretation.get("chart"),
                "agents_invoked": 4  
            }
            
        except Exception as e:
            print(f"-> Unhandled System Exception: {e}")
            return self.fallback.handle_error("system_error", str(e), user_query)

# Quick test block
if __name__ == "__main__":
    pipeline = NL2SQLPipeline()
    
    # Test 1: General Chat
    chat_result = pipeline.process_query("Hello! Who built you?")
    print(f"\nFinal Output (Chat):\n{chat_result['content']}")
    
    # Test 2: Database Query
    db_result = pipeline.process_query("How many patients are currently active in the hospital?")
    print(f"\nFinal Output (DB Query):")
    print(f"Summary: {db_result.get('content')}")
    print(f"SQL Used: {db_result.get('sql', 'N/A')}")