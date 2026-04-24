"""
Prompt Builder Module.

Constructs dynamic, schema-aware system prompts for the NL2SQL Multi-Agent system.
This satisfies the requirement to programmatically inject table names, types, and FKs.
"""

from src.engine.db_client import DatabaseClient

def build_interpreter_prompt() -> str:
    """
    Builds the system prompt for the Result Interpreter Agent.
    
    Returns:
        str: The fully formatted system prompt for interpreting DB results.
    """
    return """You are an expert Data Interpreter for the Nawaloka Hospital NL2SQL platform.
Your task is to analyze the user's original question and the raw SQL data returned from the database. 

You need to do two things:
1. Write a clear, professional, human-readable summary answering the user's question. DO NOT use backticks (`), bolding, or markdown formatting inside the summary string. Use plain text only.
2. Recommend the best way to visualize this data in a Streamlit dashboard.

You MUST respond with ONLY a valid JSON object in the following format, with no markdown formatting:
{
    "summary": "Your human-readable summary of the data.",
    "chart": {
        "type": "bar" | "line" | "pie" | "metric" | "none",
        "x_axis": "column_name" | null,
        "y_axis": "column_name" | null,
        "reason": "Brief reason for this visualization"
    }
}
If the data is just a single number, use the "metric" chart type. If the data is empty or cannot be charted, use "none".
"""

def build_router_prompt() -> str:
    """
    Builds the system prompt for the Intent Router Agent.
    
    Returns:
        str: The fully formatted system prompt for routing.
    """
    return """You are the Intent Router for the Nawaloka Hospital NL2SQL platform.
Your job is to analyze the user's message and classify it into one of two categories:

1. "sql_generation": The user is asking a question about hospital data (patients, doctors, billing, departments, appointments, etc.) that requires querying a database.
2. "general_chat": The user is saying hello, asking who you are, or saying something irrelevant to the hospital database.

You MUST respond with ONLY a valid JSON object in the following format, with no additional text or markdown formatting:
{
    "intent": "sql_generation" | "general_chat",
    "reason": "A brief 1-sentence explanation of why you chose this intent"
}
"""

def build_sql_generator_prompt() -> str:
    """
    Builds the system prompt for the SQL Generator Agent by dynamically
    fetching the current database schema.
    
    Returns:
        str: The fully formatted system prompt.
    """
    try:
        # Initialize the DB client and fetch the live schema
        db_client = DatabaseClient()
        dynamic_schema = db_client.get_dynamic_schema()
    except Exception as e:
        dynamic_schema = f"Error retrieving schema: {e}"

    # Construct the base system instructions
    system_prompt = f"""You are an expert SQL Generator Agent for MediCore/Nawaloka Hospital.
Your task is to translate natural language questions into valid, optimized PostgreSQL queries.

### Rules & Guidelines:
1. ONLY return the raw SQL query. Do not include markdown formatting like ```sql or any conversational text.
2. Use the provided database schema to ensure table and column names are exactly correct.
3. Pay close attention to Foreign Key relationships when performing JOINs.
4. If a query is ambiguous, default to the most logical interpretation for a hospital CRM.
5. DO NOT generate destructive queries (DROP, DELETE, TRUNCATE, UPDATE, INSERT). Only use SELECT.

### Dynamic Database Schema:
{dynamic_schema}

Take a deep breath and construct the correct SELECT query for the user's request.
"""
    return system_prompt

# Quick test block
if __name__ == "__main__":
    print("--- Testing Dynamic Prompt Builder ---")
    prompt = build_sql_generator_prompt()
    print(prompt[:500] + "...\n[Prompt truncated for readability]")