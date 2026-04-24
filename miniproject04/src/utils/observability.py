"""
Observability Module.
Initializes the Langfuse client for manual tracing if decorators are not used.
"""
import os
from langfuse import Langfuse
from dotenv import load_dotenv

load_dotenv()

def get_langfuse_client() -> Langfuse:
    """Returns a configured Langfuse client instance."""
    return Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    )