"""
Database Client Module.

Handles database connections and dynamic schema extraction 
to prevent hardcoded schemas in the NL2SQL engine.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()

class DatabaseClient:
    """Client for executing queries and extracting dynamic database schemas."""

    def __init__(self, connection_string: str = None):
        """
        Initializes the database connection.
        
        Args:
            connection_string (str): Optional override for the DB connection string.
        """
        self.db_url = connection_string or os.getenv("DATABASE_URL") or os.getenv("DB_CONNECTION_STRING")
        if not self.db_url:
            raise ValueError("DATABASE_URL is missing from .env file.")
        
        try:
            self.engine = create_engine(self.db_url)
            self.inspector = inspect(self.engine)
        except SQLAlchemyError as e:
            print(f"Failed to connect to database: {e}")
            raise

    def get_dynamic_schema(self) -> str:
        """
        Dynamically extracts table names, column types, foreign keys, AND sample data.
        """
        schema_details = []
        
        try:
            for table_name in self.inspector.get_table_names():
                schema_details.append(f"Table: {table_name}")
                
                # Get Columns and Types
                columns = self.inspector.get_columns(table_name)
                for col in columns:
                    schema_details.append(f"  - {col['name']} ({col['type']})")
                
                # Get Foreign Keys
                fks = self.inspector.get_foreign_keys(table_name)
                for fk in fks:
                    for constrained_col, referred_col in zip(fk['constrained_columns'], fk['referred_columns']):
                        schema_details.append(
                            f"  * Foreign Key: {constrained_col} -> {fk['referred_table']}.{referred_col}"
                        )
                
                # Get 2 Rows of Sample Data
                try:
                    with self.engine.connect() as conn:
                        samples = [dict(row) for row in conn.execute(text(f'SELECT * FROM "{table_name}" LIMIT 2')).mappings()]
                        if samples:
                            # Convert to string and truncate slightly to save LLM tokens
                            sample_str = str(samples)[:300] + ("..." if len(str(samples)) > 300 else "")
                            schema_details.append(f"  * Sample Data: {sample_str}")
                except Exception as e:
                    schema_details.append("  * Sample Data: [Could not extract]")

                schema_details.append("") 
                
            return "\n".join(schema_details)
            
        except SQLAlchemyError as e:
            error_msg = f"Error extracting schema: {e}"
            print(error_msg)
            return error_msg

    def execute_query(self, query: str) -> list:
        """
        Executes a safe SQL query and returns the results.
        
        Args:
            query (str): The validated SELECT query to execute.
            
        Returns:
            list: List of dictionaries representing the rows.
        """
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text(query))
                return [dict(row) for row in result.mappings()]
        except SQLAlchemyError as e:
            return [{"error": str(e)}]

# Quick test block
if __name__ == "__main__":
    try:
        client = DatabaseClient()
        print(client.get_dynamic_schema())
    except Exception as err:
        print(f"Setup Error: {err}")