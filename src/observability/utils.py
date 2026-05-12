import json
import os
from datetime import datetime
from typing import Any, Dict

def save_json(data: Dict[str, Any], filepath: str):
    """
    Saves a dictionary as a JSON file.
    Creates directories if they don't exist.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def get_timestamp() -> str:
    """Returns the current timestamp in ISO format."""
    return datetime.now().isoformat()
