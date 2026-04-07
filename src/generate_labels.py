"""
generate_labels.py — Create labels.json mapping from pokemon.json

Extracts all unique type1 values from pokemon.json and creates a mapping
to integers (sorted alphabetically for reproducibility).
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_DIR = PROJECT_ROOT / "data"
LABELS_PATH = DATA_DIR / "labels.json"


def generate_labels() -> None:
    """Generate labels.json from pokemon.json."""
    
    pokemon_path = RAW_DIR / "pokemon.json"
    
    if not pokemon_path.exists():
        print(f"❌ Error: {pokemon_path} not found")
        print("   Run: python src/picker.py first")
        return
    
    # Load pokemon data
    with pokemon_path.open(encoding="utf-8") as f:
        pokemon_list = json.load(f)
    
    # Extract unique types
    types = sorted(set(p["type1"] for p in pokemon_list))
    
    # Create mapping
    labels = {t: i for i, t in enumerate(types)}
    
    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with LABELS_PATH.open("w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created {LABELS_PATH}")
    print(f"\n  Types ({len(types)}): {', '.join(types)}")
    print(f"\n  Mapping example:")
    for t, i in list(labels.items())[:3]:
        print(f"    {t}: {i}")
    print(f"    ...")


if __name__ == "__main__":
    generate_labels()
