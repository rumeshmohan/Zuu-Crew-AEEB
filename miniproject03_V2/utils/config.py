import os
import yaml
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

load_dotenv()

_ENV_KEY_MAP = {
    "openai":      "OPENAI_API_KEY",
    "openrouter":  "OPENROUTER_API_KEY",
    "groq":        "GROQ_API_KEY",
    "google":      "GEMINI_API_KEY",
    "gemini":      "GEMINI_API_KEY",
    "cohere":      "COHERE_API_KEY",
}


class ConfigManager:
    """Loads and exposes application settings from params.yaml and model.yaml."""

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self._models: Dict[str, Any] = {}
        self._load_configs()

    def _load_configs(self) -> None:
        """Read params.yaml and model.yaml from the config/ directory into memory."""
        config_dir = Path(__file__).resolve().parent.parent / "config"

        params_path = config_dir / "params.yaml"
        if params_path.exists():
            with open(params_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}

        models_path = config_dir / "model.yaml"
        if models_path.exists():
            with open(models_path, "r", encoding="utf-8") as f:
                self._models = yaml.safe_load(f) or {}

    def get(self, key_path: str, default: Any = None) -> Any:
        """Return the config value at the dot-separated key_path, or default if not found."""
        value = self.config
        for key in key_path.split("."):
            if not isinstance(value, dict):
                return default
            value = value.get(key)
            if value is None:
                return default
        return value

    def get_model(self, provider: str, tier: str, is_embedding: bool = False) -> str:
        """Return the model identifier for the given provider, tier, and type."""
        model_type = "embedding" if is_embedding else "chat"
        try:
            return self._models[provider][model_type][tier]
        except KeyError:
            return "gpt-4o-mini"


def get_api_key(service: str) -> str:
    """Return the API key for the given service from environment variables."""
    env_var = _ENV_KEY_MAP.get(service.lower(), "")
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"API key not found for {service}. Check .env!")
    return api_key


_config_instance: ConfigManager | None = None


def get_config() -> ConfigManager:
    """Return the global ConfigManager singleton, creating it on first call."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager()
    return _config_instance