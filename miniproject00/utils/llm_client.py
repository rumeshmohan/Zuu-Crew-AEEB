import os
from pathlib import Path
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

# SDK Imports
from groq import Groq
from google import genai
from openai import OpenAI
from mistralai import Mistral
import cohere

from .config_loader import get_default_temperature, get_default_max_tokens
from .router import pick_model

# Path-aware loading for .env
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class LLMClient:
    def __init__(self, provider: str, technique: str):
        self.provider = provider.lower()
        self.model = pick_model(self.provider, technique)
        
        # Initialize the specific provider
        if self.provider == "groq":
            self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        elif self.provider == "google":
            self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        elif self.provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "cohere":
            self.client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        elif self.provider == "mistral":
            self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # Increased wait time to handle Mistral/Groq rate limits (Status 429)
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=5, max=60))
    def chat(self, messages: list, task_type: str = "classification", temperature: float = None, max_tokens: int = None):
        """Executes a chat completion across different providers with a unified output."""
        temp = temperature if temperature is not None else get_default_temperature(task_type)
        max_tok = max_tokens if max_tokens is not None else get_default_max_tokens(task_type)

        # 1. OpenAI & Groq (Standard Completions)
        if self.provider in ["openai", "groq"]:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok
            )
            return {
                "text": response.choices[0].message.content.strip(),
                "usage": response.usage
            }

        # 2. Mistral (Handles Multi-part Content and Reasoning chunks)
        elif self.provider == "mistral":
            response = self.client.chat.complete(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok
            )
            content = response.choices[0].message.content
            # Handle list-based responses (TextChunks/ThinkChunks)
            if isinstance(content, list):
                text_out = "".join([chunk.text for chunk in content if hasattr(chunk, 'text')])
            else:
                text_out = content
            return {
                "text": text_out.strip(),
                "usage": response.usage
            }

        # 3. Cohere V2
        elif self.provider == "cohere":
            response = self.client.chat(
                model=self.model,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok
            )
            return {
                "text": response.message.content[0].text.strip(),
                "usage": response.usage
            }

        # 4. Google Gemini (Vertex/GenAI API)
        elif self.provider == "google":
            # Extract only the last user message for simple prompt processing
            prompt_content = messages[-1]["content"]
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt_content,
                config={'temperature': temp, 'max_output_tokens': max_tok}
            )
            return {
                "text": response.text.strip(),
                "usage": response.usage_metadata
            }

        raise ValueError(f"Provider {self.provider} logic not implemented.")