import logging
from openai import OpenAI
from src.utils.config import get_api_key, get_config

logger = logging.getLogger(__name__)

# Base URLs for the OpenAI-compatible providers
_BASE_URLS = {
    "ollama":     "http://localhost:11434/v1",
    "groq":       "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek":   "https://api.deepseek.com",
    "mistral":    "https://api.mistral.ai/v1",
}

class LLMProvider:
    """Wraps API clients for 8 configured LLM providers, using native SDKs where appropriate."""

    def __init__(self, tier: str = "general"):
        """Initialise the client and resolve the model for the given tier."""
        self.config = get_config()
        self.provider = self.config.get("provider.default", "openai").lower()

        raw_model = self.config.get_model(self.provider, tier)
        self.model = raw_model.split("/")[-1] if "/" in raw_model else raw_model

        # 1. Native Anthropic (Claude) Setup
        if self.provider in ["anthropic", "claude"]:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=get_api_key("anthropic"))
            self.client_type = "anthropic"

        # 2. Native Google (Gemini) Setup
        elif self.provider in ["google", "gemini"]:
            from google import genai
            self.client_type = "gemini"
            self.client = genai.Client(api_key=get_api_key("gemini"))

        # 3. OpenAI-Compatible Setup (OpenAI, Groq, DeepSeek, Ollama, OpenRouter, Mistral, Cohere)
        else:
            self.client_type = "openai"
            
            if self.provider == "ollama":
                base_url = _BASE_URLS["ollama"]
                api_key = "ollama"
            elif self.provider in _BASE_URLS:
                base_url = _BASE_URLS[self.provider]
                api_key = get_api_key(self.provider)
            else:
                base_url = None  # Defaults to official OpenAI URL
                api_key = get_api_key("openai")

            self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=120)

    def generate(self, prompt: str, system_prompt: str = None) -> str:
        """Send a prompt to the LLM and return the text response."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        return self.generate_messages(messages)

    def generate_messages(self, messages: list) -> str:
        """Send a full list of message dictionaries to the LLM for multi-turn loops."""
        try:
            temperature = self.config.get("llm.temperature", 0.0)
            max_tokens = self.config.get("llm.max_tokens", 1024)

            # Execution for Anthropic (Claude)
            if self.client_type == "anthropic":
                system_text = ""
                anthropic_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_text += msg["content"] + "\n"
                    else:
                        anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
                
                response = self.client.messages.create(
                    model=self.model,
                    system=system_text.strip() if system_text else None,
                    messages=anthropic_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.content[0].text

            # Execution for Google (Gemini)
            elif self.client_type == "gemini":
                from google.genai import types
                
                system_text = ""
                gemini_messages = []
                for msg in messages:
                    if msg["role"] == "system":
                        system_text += msg["content"] + "\n"
                    else:
                        role = "model" if msg["role"] == "assistant" else "user"
                        gemini_messages.append(types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])]))
                
                config = types.GenerateContentConfig(
                    system_instruction=system_text.strip() if system_text else None,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                )
                
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=gemini_messages,
                    config=config
                )
                return response.text

            # Execution for all OpenAI-Compatible Providers
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"❌ API Error on model {self.model} via {self.provider}: {e}")
            raise

def get_llm(tier: str = "general") -> LLMProvider:
    """Return a configured LLMProvider for the given tier."""
    return LLMProvider(tier=tier)