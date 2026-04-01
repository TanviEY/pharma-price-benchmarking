import json
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"

def load_molecule_mapping() -> Dict:
    config_path = CONFIG_DIR / "molecule_mapping.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

MOLECULE_MAPPING = load_molecule_mapping()

