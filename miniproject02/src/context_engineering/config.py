"""
Application configuration - loads from YAML config files.

Configuration is loaded from config/config.yaml.
Secrets (API keys) live ONLY in .env and are loaded via os.getenv().
Supports separate providers for embeddings and chat/LLM.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import os
import yaml

# Project root is 3 levels up from this file (src/context_engineering/)
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config"


# ── YAML Helpers ─────────────────────────────────────────────────────────────

def _load_yaml(filename: str) -> Dict[str, Any]:
    """Load a YAML config file from the config directory.

    Args:
        filename: Name of the YAML file (e.g. 'config.yaml').

    Returns:
        Parsed dictionary, or empty dict if file is missing.
    """
    filepath = _CONFIG_DIR / filename
    if not filepath.exists():
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get_nested(d: Dict, *keys, default=None):
    """Safely traverse a nested dictionary.

    Args:
        d: Source dictionary.
        *keys: Sequence of keys to traverse.
        default: Value returned when any key is absent.

    Returns:
        The value at the nested path, or *default*.
    """
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key, default)
        else:
            return default
    return d if d is not None else default


_CONFIG = _load_yaml("config.yaml")


# ── Embedding ─────────────────────────────────────────────────────────────────
# All values come from config.yaml, never from environment variables.

EMBEDDING_PROVIDER = _get_nested(_CONFIG, "embedding", "provider", default="huggingface")
EMBEDDING_TIER = _get_nested(_CONFIG, "embedding", "tier", default="small")
EMBEDDING_BATCH_SIZE = _get_nested(_CONFIG, "embedding", "batch_size", default=100)
EMBEDDING_SHOW_PROGRESS = _get_nested(_CONFIG, "embedding", "show_progress", default=False)


# ── Chat / LLM ────────────────────────────────────────────────────────────────
# All values come from config.yaml, never from environment variables.

CHAT_PROVIDER = _get_nested(_CONFIG, "llm", "provider", default="groq")
CHAT_TIER = _get_nested(_CONFIG, "llm", "tier", default="general")
LLM_TEMPERATURE = _get_nested(_CONFIG, "llm", "temperature", default=0.0)
LLM_MAX_TOKENS = _get_nested(_CONFIG, "llm", "max_tokens", default=2000)
LLM_STREAMING = _get_nested(_CONFIG, "llm", "streaming", default=False)
OPENROUTER_BASE_URL = _get_nested(_CONFIG, "llm", "openrouter_base_url",
                                   default="https://openrouter.ai/api/v1")


# ── Model Resolution (tier → model name) ─────────────────────────────────────

_MODELS_CONFIG = _load_yaml("models.yaml")


def _resolve_chat_model(provider: str, tier: str) -> str:
    """Resolve a chat tier to the actual model name from models.yaml.

    Args:
        provider: LLM provider key (e.g. 'groq').
        tier: Tier label (e.g. 'general').

    Returns:
        Model name string.

    Raises:
        ValueError: If provider or tier is not present in models.yaml.
    """
    provider = provider.lower()
    if provider not in _MODELS_CONFIG:
        raise ValueError(f"Provider '{provider}' not found in models.yaml")
    if "chat" not in _MODELS_CONFIG[provider]:
        raise ValueError(f"Provider '{provider}' has no chat models in models.yaml")
    if tier not in _MODELS_CONFIG[provider]["chat"]:
        raise ValueError(f"Tier '{tier}' not found for provider '{provider}' in models.yaml")
    return _MODELS_CONFIG[provider]["chat"][tier]


def _resolve_embedding_model(provider: str, tier: str) -> str:
    """Resolve an embedding tier to the actual model name from models.yaml.

    Args:
        provider: Embedding provider key (e.g. 'huggingface').
        tier: Tier label (e.g. 'small').

    Returns:
        Model name string.

    Raises:
        ValueError: If provider or tier is not present in models.yaml.
    """
    provider = provider.lower()
    if provider not in _MODELS_CONFIG:
        raise ValueError(f"Provider '{provider}' not found in models.yaml")
    if "embedding" not in _MODELS_CONFIG[provider]:
        raise ValueError(f"Provider '{provider}' has no embedding models in models.yaml")
    if tier not in _MODELS_CONFIG[provider]["embedding"]:
        raise ValueError(f"Tier '{tier}' not found for provider '{provider}' in models.yaml")
    return _MODELS_CONFIG[provider]["embedding"][tier]


CHAT_MODEL = _resolve_chat_model(CHAT_PROVIDER, CHAT_TIER)
EMBEDDING_MODEL = _resolve_embedding_model(EMBEDDING_PROVIDER, EMBEDDING_TIER)

PROVIDER = CHAT_PROVIDER  # Deprecated – use CHAT_PROVIDER instead


# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = _PROJECT_ROOT / _get_nested(_CONFIG, "paths", "data_dir", default="data")
VECTOR_DIR = _PROJECT_ROOT / _get_nested(_CONFIG, "paths", "vector_store", default="data/vectorstore")
MARKDOWN_DIR = _PROJECT_ROOT / _get_nested(_CONFIG, "paths", "markdown_dir", default="data/primelands_markdown")
CACHE_DIR = _PROJECT_ROOT / _get_nested(_CONFIG, "paths", "cache_dir", default="data/cag_cache")
LOGS_DIR = _PROJECT_ROOT / _get_nested(_CONFIG, "paths", "logs_dir", default="logs")
CORPUS_FILE = _PROJECT_ROOT / _get_nested(_CONFIG, "paths", "corpus_file", default="data/primelands_corpus.jsonl")
CRAWL_OUT_DIR = DATA_DIR


# ── Qdrant ────────────────────────────────────────────────────────────────────

QDRANT_COLLECTIONS = _get_nested(_CONFIG, "qdrant", "collections", default=[
    "primelands_semantic",
    "primelands_fixed",
    "primelands_sliding",
    "primelands_parent_child",
    "primelands_late_chunk",
])
QDRANT_DEFAULT_COLLECTION = _get_nested(_CONFIG, "qdrant", "default_collection",
                                         default="primelands_semantic")


# ── Chunking ──────────────────────────────────────────────────────────────────

FIXED_CHUNK_SIZE = _get_nested(_CONFIG, "chunking", "fixed", "chunk_size", default=800)
FIXED_CHUNK_OVERLAP = _get_nested(_CONFIG, "chunking", "fixed", "chunk_overlap", default=100)

SEMANTIC_MAX_CHUNK_SIZE = _get_nested(_CONFIG, "chunking", "semantic", "max_chunk_size", default=1000)
SEMANTIC_MIN_CHUNK_SIZE = _get_nested(_CONFIG, "chunking", "semantic", "min_chunk_size", default=200)

SLIDING_WINDOW_SIZE = _get_nested(_CONFIG, "chunking", "sliding", "window_size", default=512)
SLIDING_STRIDE_SIZE = _get_nested(_CONFIG, "chunking", "sliding", "stride_size", default=256)

PARENT_CHUNK_SIZE = _get_nested(_CONFIG, "chunking", "parent_child", "parent_size", default=1200)
CHILD_CHUNK_SIZE = _get_nested(_CONFIG, "chunking", "parent_child", "child_size", default=250)
CHILD_OVERLAP = _get_nested(_CONFIG, "chunking", "parent_child", "child_overlap", default=50)

LATE_CHUNK_BASE_SIZE = _get_nested(_CONFIG, "chunking", "late", "base_size", default=1000)
LATE_CHUNK_SPLIT_SIZE = _get_nested(_CONFIG, "chunking", "late", "split_size", default=300)
LATE_CHUNK_CONTEXT_WINDOW = _get_nested(_CONFIG, "chunking", "late", "context_window", default=150)


# ── Retrieval ─────────────────────────────────────────────────────────────────

TOP_K_RESULTS = _get_nested(_CONFIG, "retrieval", "top_k", default=4)
SIMILARITY_THRESHOLD = _get_nested(_CONFIG, "retrieval", "similarity_threshold", default=0.7)


# ── CAG ───────────────────────────────────────────────────────────────────────

CAG_CACHE_MAX_SIZE = _get_nested(_CONFIG, "cag", "max_cache_size", default=1000)
CAG_SIMILARITY_THRESHOLD = _get_nested(_CONFIG, "cag", "similarity_threshold", default=0.90)
CAG_HISTORY_TTL_HOURS = _get_nested(_CONFIG, "cag", "history_ttl_hours", default=24)


# ── CRAG ──────────────────────────────────────────────────────────────────────

CRAG_CONFIDENCE_THRESHOLD = _get_nested(_CONFIG, "crag", "confidence_threshold", default=0.6)
CRAG_EXPANDED_K = _get_nested(_CONFIG, "crag", "expanded_k", default=8)


# ── Crawling ──────────────────────────────────────────────────────────────────

CRAWL_BASE_URL = _get_nested(_CONFIG, "crawling", "base_url", default="https://www.primelands.lk")
CRAWL_MAX_DEPTH = _get_nested(_CONFIG, "crawling", "max_depth", default=2)
CRAWL_DELAY_SECONDS = _get_nested(_CONFIG, "crawling", "delay_seconds", default=2.0)
CRAWL_MAX_PAGES = _get_nested(_CONFIG, "crawling", "max_pages", default=150)
CRAWL_START_PATHS = _get_nested(_CONFIG, "crawling", "start_paths", default=[])
CRAWL_EXCLUDE_PATTERNS = _get_nested(_CONFIG, "crawling", "exclude_patterns", default=[])


# ── Logging ───────────────────────────────────────────────────────────────────

LOGGING_ENABLED = _get_nested(_CONFIG, "logging", "enabled", default=True)
LOGGING_LEVEL = _get_nested(_CONFIG, "logging", "level", default="INFO")
LOGGING_LOG_TOKENS = _get_nested(_CONFIG, "logging", "log_tokens", default=True)
LOGGING_LOG_LATENCY = _get_nested(_CONFIG, "logging", "log_latency", default=True)


# ── Helper Functions ──────────────────────────────────────────────────────────

def get_api_key(provider: Optional[str] = None) -> Optional[str]:
    """Get the API key for a provider from environment variables.

    Args:
        provider: Provider name. Defaults to CHAT_PROVIDER when None.

    Returns:
        API key string, or None if the provider needs no key or key is unset.
    """
    if provider is None:
        provider = CHAT_PROVIDER

    key_map = {
        "openrouter": "OPENROUTER_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GEMINI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "cohere": "COHERE_API_KEY",
        "huggingface": None,  # Local model – no API key required
    }

    env_var = key_map.get(provider.lower())
    if env_var is None:
        return None

    return os.getenv(env_var)


def get_embedding_api_key() -> Optional[str]:
    """Get the API key for the active embedding provider."""
    return get_api_key(EMBEDDING_PROVIDER)


def get_chat_api_key() -> Optional[str]:
    """Get the API key for the active chat provider."""
    return get_api_key(CHAT_PROVIDER)


def validate() -> None:
    """Validate configuration and create required directories.

    Raises:
        ValueError: If a required API key is missing from the environment.
        OSError: If a required directory cannot be created.
    """
    if EMBEDDING_PROVIDER.lower() not in ["huggingface"]:
        embedding_key = get_embedding_api_key()
        if not embedding_key:
            key_name = f"{EMBEDDING_PROVIDER.upper()}_API_KEY"
            if EMBEDDING_PROVIDER.lower() in ["google", "gemini"]:
                key_name = "GEMINI_API_KEY"
            raise ValueError(
                f"❌ Missing required secret for embedding provider: {key_name}\n"
                f"   Current provider in config.yaml: {EMBEDDING_PROVIDER}\n"
                f"   Please add {key_name} to your .env file\n"
                f"   OR change embedding.provider in config.yaml to 'huggingface' (free, no API key)"
            )

    chat_key = get_chat_api_key()
    if not chat_key:
        key_name = f"{CHAT_PROVIDER.upper()}_API_KEY"
        if CHAT_PROVIDER.lower() in ["google", "gemini"]:
            key_name = "GEMINI_API_KEY"
        raise ValueError(
            f"❌ Missing required secret for chat provider: {key_name}\n"
            f"   Current provider in config.yaml: {CHAT_PROVIDER}\n"
            f"   Please add {key_name} to your .env file"
        )

    required_dirs = [DATA_DIR, VECTOR_DIR, MARKDOWN_DIR, CACHE_DIR, LOGS_DIR]
    for dir_path in required_dirs:
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise OSError(f"❌ Cannot create directory {dir_path}: {e}")


def dump() -> None:
    """Print all active non-secret configuration values for debugging."""
    print("\n" + "=" * 60)
    print("CONFIGURATION (NON-SECRETS ONLY)")
    print("=" * 60)

    print("\n🎯 EMBEDDING Configuration:")
    print(f"   Provider: {EMBEDDING_PROVIDER}")
    print(f"   Tier: {EMBEDDING_TIER}")
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   Batch Size: {EMBEDDING_BATCH_SIZE}")
    api_key = get_embedding_api_key()
    if EMBEDDING_PROVIDER == "huggingface":
        print(f"   API Key: ❌ Not needed (local)")
    else:
        print(f"   API Key: {'✅ Set' if api_key else '❌ Missing'}")

    print("\n💬 CHAT/LLM Configuration:")
    print(f"   Provider: {CHAT_PROVIDER}")
    print(f"   Tier: {CHAT_TIER}")
    print(f"   Model: {CHAT_MODEL}")
    print(f"   Temperature: {LLM_TEMPERATURE}")
    print(f"   Max Tokens: {LLM_MAX_TOKENS}")
    print(f"   Streaming: {LLM_STREAMING}")
    api_key = get_chat_api_key()
    print(f"   API Key: {'✅ Set' if api_key else '❌ Missing'}")

    print("\n📁 Directories:")
    print(f"   Data Root: {DATA_DIR}")
    print(f"   Vector Store: {VECTOR_DIR}")
    print(f"   Markdown: {MARKDOWN_DIR}")
    print(f"   Cache: {CACHE_DIR}")
    print(f"   Logs: {LOGS_DIR}")
    print(f"   Corpus File: {CORPUS_FILE}")

    print("\n🔧 Chunking:")
    print(f"   Fixed Size: {FIXED_CHUNK_SIZE} tokens")
    print(f"   Fixed Overlap: {FIXED_CHUNK_OVERLAP} tokens")
    print(f"   Sliding Window: {SLIDING_WINDOW_SIZE} tokens")
    print(f"   Sliding Stride: {SLIDING_STRIDE_SIZE} tokens")
    print(f"   Parent-Child: {CHILD_CHUNK_SIZE} → {PARENT_CHUNK_SIZE} tokens")
    print(f"   Late Chunk: {LATE_CHUNK_BASE_SIZE} → {LATE_CHUNK_SPLIT_SIZE} tokens")

    print("\n🔍 Retrieval:")
    print(f"   Top-K Results: {TOP_K_RESULTS}")
    print(f"   Similarity Threshold: {SIMILARITY_THRESHOLD}")

    print("\n💾 CAG:")
    print(f"   Max Cache Size: {CAG_CACHE_MAX_SIZE}")
    print(f"   Similarity Threshold: {CAG_SIMILARITY_THRESHOLD}")
    print(f"   History TTL: {CAG_HISTORY_TTL_HOURS}h")

    print("\n🎯 CRAG:")
    print(f"   Confidence Threshold: {CRAG_CONFIDENCE_THRESHOLD}")
    print(f"   Expanded K: {CRAG_EXPANDED_K}")

    print("\n🕷️ Crawling:")
    print(f"   Base URL: {CRAWL_BASE_URL}")
    print(f"   Max Depth: {CRAWL_MAX_DEPTH}")
    print(f"   Max Pages: {CRAWL_MAX_PAGES}")
    print(f"   Delay: {CRAWL_DELAY_SECONDS}s")
    print(f"   Start Paths: {len(CRAWL_START_PATHS)} configured")
    print(f"   Exclude Patterns: {len(CRAWL_EXCLUDE_PATTERNS)} configured")

    print("\n💡 To change providers:")
    print(f"   Edit config/config.yaml (embedding.provider and llm.provider)")
    print(f"   Models are mapped in config/models.yaml")
    print(f"   Add API keys to .env file")

    print("\n" + "=" * 60 + "\n")


def get_config() -> Dict[str, Any]:
    """Return the full raw configuration dictionary."""
    return _CONFIG


# ── Deprecated Functions ──────────────────────────────────────────────────────

def get_chat_tier(provider: Optional[str] = None, tier: Optional[str] = None) -> str:
    """Deprecated: use the CHAT_MODEL constant instead."""
    print("⚠️  Warning: get_chat_model() is deprecated. Use CHAT_MODEL constant instead.")
    return CHAT_TIER


def get_embedding_model(provider: Optional[str] = None, tier: str = "default") -> str:
    """Deprecated: use the EMBEDDING_MODEL constant instead."""
    print("⚠️  Warning: get_embedding_model() is deprecated. Use EMBEDDING_MODEL constant instead.")
    return EMBEDDING_MODEL


def load_faqs() -> list:
    """Load FAQ questions from config/faqs.yaml.

    Returns:
        Flat list of FAQ question strings across all categories.
    """
    try:
        faqs_config = _load_yaml("faqs.yaml")
        if not faqs_config or not isinstance(faqs_config, dict):
            print(f"⚠️  faqs.yaml is empty or not a valid dict. Got: {type(faqs_config)}")
            return []
        all_faqs = []
        for category, questions in faqs_config.items():
            if isinstance(questions, list):
                all_faqs.extend(q for q in questions if isinstance(q, str))
        print(f"✅ Loaded {len(all_faqs)} FAQs from faqs.yaml")
        return all_faqs
    except Exception as e:
        print(f"❌ Failed to load faqs.yaml: {e}")
        return []


KNOWN_FAQS = load_faqs()