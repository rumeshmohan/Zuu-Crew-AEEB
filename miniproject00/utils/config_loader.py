import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass

@dataclass
class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

_config: Optional[Config] = None

def load_config():
    global _config
    path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(path, "r") as f:
        _config = Config(yaml.safe_load(f))
    return _config

def get_config():
    global _config
    if _config is None: load_config()
    return _config

def get_default_temperature(task_type: Optional[str] = None) -> float:
    if task_type:
        temp = get_config().get(f"defaults.by_task.{task_type}.temperature")
        if temp is not None: return temp
    return get_config().get("defaults.temperature", 0.2)


def get_default_max_tokens(task_type: Optional[str] = None) -> int:
    if task_type:
        max_tok = get_config().get(f"defaults.by_task.{task_type}.max_tokens")
        if max_tok is not None:
            return max_tok
    return get_config().get("defaults.max_tokens", 1000)