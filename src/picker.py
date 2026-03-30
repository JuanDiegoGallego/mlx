"""
picker.py — Data collection script for PokéAPI.

Downloads Pokémon metadata (name, id, type1, type2) and front sprites for
Pokémon IDs 1 through 1025, saving results to data/raw/.
"""

import json
import time
from pathlib import Path
from typing import Optional

import requests

# Constants
MAX_POKEMON_ID = 1025
REQUEST_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3
BASE_URL = "https://pokeapi.co/api/v2/pokemon"
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def get_pokemon_data(session: requests.Session, pokemon_id: int) -> Optional[dict]:
    """Fetch Pokémon data from PokéAPI with retry logic.

    Args:
        session: requests.Session for connection reuse.
        pokemon_id: Pokédex number to fetch.

    Returns:
        Parsed JSON response dict, or None if all retries fail.
    """
    url = f"{BASE_URL}/{pokemon_id}/"
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  Attempt {attempt + 1}/{MAX_RETRIES} failed for ID {pokemon_id}: {e}. "
                  f"Retrying in {wait}s...")
            time.sleep(wait)
    return None


def download_sprite(session: requests.Session, sprite_url: str, dest_path: Path) -> bool:
    """Download a sprite image to disk.

    Args:
        session: requests.Session for connection reuse.
        sprite_url: URL of the sprite PNG.
        dest_path: Local path to save the file.

    Returns:
        True on success, False on failure.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(sprite_url, timeout=10)
            response.raise_for_status()
            dest_path.write_bytes(response.content)
            return True
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  Sprite download attempt {attempt + 1}/{MAX_RETRIES} failed: {e}. "
                  f"Retrying in {wait}s...")
            time.sleep(wait)
    return False


def extract_metadata(data: dict) -> dict:
    """Extract relevant fields from a PokéAPI Pokémon response.

    Args:
        data: Full PokéAPI response for a single Pokémon.

    Returns:
        Dict with keys: name, id, type1, type2 (type2 is None if single-type).
    """
    name = data["name"]
    pokemon_id = data["id"]

    type1 = None
    type2 = None
    for entry in data["types"]:
        if entry["slot"] == 1:
            type1 = entry["type"]["name"]
        elif entry["slot"] == 2:
            type2 = entry["type"]["name"]

    return {"name": name, "id": pokemon_id, "type1": type1, "type2": type2}


def collect_all_pokemon() -> None:
    """Main routine: fetch metadata and sprites for all Pokémon IDs 1–1025.

    Saves:
        data/raw/{id}.png  — front sprite for each Pokémon
        data/raw/pokemon.json — array of metadata dicts
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    failed_ids: list[int] = []
    skipped_null_sprite: list[int] = []
    pokemon_list: list[dict] = []

    session = requests.Session()

    for pokemon_id in range(1, MAX_POKEMON_ID + 1):
        sprite_path = RAW_DIR / f"{pokemon_id}.png"

        # Fetch metadata from API regardless (needed for pokemon.json)
        data = get_pokemon_data(session, pokemon_id)
        if data is None:
            print(f"  FAILED: Could not fetch ID {pokemon_id} after {MAX_RETRIES} attempts.")
            failed_ids.append(pokemon_id)
            time.sleep(REQUEST_DELAY)
            continue

        metadata = extract_metadata(data)
        sprite_url = data["sprites"]["front_default"]

        if sprite_url is None:
            print(f"  SKIPPED: ID {pokemon_id} ({metadata['name']}) has no front_default sprite.")
            skipped_null_sprite.append(pokemon_id)
            time.sleep(REQUEST_DELAY)
            continue

        pokemon_list.append(metadata)

        # Download sprite only if not already on disk
        if sprite_path.exists():
            pass  # already downloaded, skip
        else:
            success = download_sprite(session, sprite_url, sprite_path)
            if not success:
                print(f"  FAILED: Sprite download for ID {pokemon_id}.")
                failed_ids.append(pokemon_id)
                pokemon_list.pop()  # remove metadata for failed sprite
                time.sleep(REQUEST_DELAY)
                continue

        if pokemon_id % 50 == 0:
            print(f"Fetched {pokemon_id}/{MAX_POKEMON_ID}...")

        time.sleep(REQUEST_DELAY)

    # Save metadata
    json_path = RAW_DIR / "pokemon.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(pokemon_list, f, indent=2, ensure_ascii=False)

    print(f"\nDone. {len(pokemon_list)} Pokémon saved to {json_path}")

    if skipped_null_sprite:
        print(f"Skipped (null sprite): {skipped_null_sprite}")
    if failed_ids:
        print(f"Failed IDs: {failed_ids}")
    else:
        print("No failures.")


if __name__ == "__main__":
    collect_all_pokemon()
