import yaml
from pathlib import Path
from typing import Literal, Optional
from .config_loader import get_config

def pick_model(provider: str, technique: str) -> str:
    path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(path, "r") as f:
        models = yaml.safe_load(f)
    
    # Logic: Auto-route reasoning techniques
    if any(x in technique.lower() for x in ["cot", "tot", "reason"]):
        tier = "reason"
    else:
        tier = "general"
    return models[provider][tier]