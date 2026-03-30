"""
processer.py — Feature extraction pipeline for Pokémon sprites.

Reads data/raw/pokemon.json and corresponding PNG sprites, extracts 42 color
features per sprite (K-means dominant colors, HSV statistics, hue histogram),
and saves the result to data/processed/features.csv.
"""

import json
import math
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

# Allow Pillow to open PNGs with embedded ICC profiles that would otherwise
# raise "cannot identify image file" (affects e.g. Pokémon #678, #696).
ImageFile.LOAD_TRUNCATED_IMAGES = True
from scipy.stats import circstd
from sklearn.cluster import KMeans

# Constants
K_CLUSTERS = 5
NUM_HUE_BINS = 12
HUE_BIN_WIDTH = 360 // NUM_HUE_BINS  # 30°
MIN_PIXELS = 50
KMEANS_SEED = 42

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LABELS_PATH = PROJECT_ROOT / "data" / "labels.json"


# ---------------------------------------------------------------------------
# Step 0: Load and mask
# ---------------------------------------------------------------------------

def load_sprite_pixels(sprite_path: Path) -> Optional[np.ndarray]:
    """Load a PNG sprite and return non-transparent RGB pixels.

    Args:
        sprite_path: Path to the PNG file (must have RGBA mode).

    Returns:
        (N, 3) uint8 array of RGB values for pixels with alpha > 0,
        or None if the sprite cannot be loaded.
    """
    try:
        img = Image.open(sprite_path).convert("RGBA")
    except Exception as e:
        warnings.warn(f"Cannot open {sprite_path}: {e}")
        return None

    arr = np.array(img)  # (H, W, 4)
    alpha = arr[:, :, 3]
    mask = alpha > 0
    rgb = arr[:, :, :3][mask]  # (N, 3)
    return rgb


# ---------------------------------------------------------------------------
# Step 1: Convert to HSV
# ---------------------------------------------------------------------------

def rgb_to_hsv_array(rgb: np.ndarray) -> np.ndarray:
    """Convert (N, 3) uint8 RGB array to (N, 3) float HSV array.

    Returns:
        Array with H in [0, 360), S in [0, 1], V in [0, 1].
    """
    rgb_float = rgb.astype(np.float32) / 255.0  # (N, 3) in [0,1]

    r, g, b = rgb_float[:, 0], rgb_float[:, 1], rgb_float[:, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    # Value
    v = cmax

    # Saturation (suppress divide-by-zero — handled by np.where condition)
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(cmax == 0, 0.0, delta / cmax)

    # Hue
    h = np.zeros(len(r), dtype=np.float32)
    mask_r = (cmax == r) & (delta != 0)
    mask_g = (cmax == g) & (delta != 0)
    mask_b = (cmax == b) & (delta != 0)

    h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    return np.stack([h, s, v], axis=1)


# ---------------------------------------------------------------------------
# Step 2 & 3: K-means dominant colors
# ---------------------------------------------------------------------------

def compute_kmeans_features(hsv: np.ndarray) -> dict:
    """Run K-means on HSV pixels and return dominant color features.

    Uses circular hue encoding ([sin, cos, S, V]) so K-means handles the
    hue wrap-around correctly.

    Args:
        hsv: (N, 3) float array with H in [0, 360), S and V in [0, 1].

    Returns:
        Dict with keys dom{i}_h, dom{i}_s, dom{i}_v, dom{i}_prop for i=1..5,
        sorted by cluster proportion descending.
    """
    h_rad = 2 * math.pi * hsv[:, 0] / 360.0
    hue_sin = np.sin(h_rad)
    hue_cos = np.cos(h_rad)
    s = hsv[:, 1]
    v = hsv[:, 2]

    X = np.stack([hue_sin, hue_cos, s, v], axis=1)

    km = KMeans(n_clusters=K_CLUSTERS, random_state=KMEANS_SEED, n_init=10)
    labels = km.fit_predict(X)

    n_pixels = len(hsv)
    clusters = []
    for cluster_id in range(K_CLUSTERS):
        mask = labels == cluster_id
        count = mask.sum()
        centroid = km.cluster_centers_[cluster_id]
        c_sin, c_cos, c_s, c_v = centroid

        # Recover hue from sin/cos
        c_h = math.degrees(math.atan2(c_sin, c_cos)) % 360

        clusters.append({
            "h": c_h,
            "s": float(c_s),
            "v": float(c_v),
            "prop": count / n_pixels,
        })

    # Sort by proportion descending
    clusters.sort(key=lambda x: x["prop"], reverse=True)

    features: dict = {}
    for i, cluster in enumerate(clusters, start=1):
        features[f"dom{i}_h"] = cluster["h"]
        features[f"dom{i}_s"] = cluster["s"]
        features[f"dom{i}_v"] = cluster["v"]
        features[f"dom{i}_prop"] = cluster["prop"]

    return features


# ---------------------------------------------------------------------------
# Step 4: Distribution statistics
# ---------------------------------------------------------------------------

def compute_stat_features(hsv: np.ndarray) -> dict:
    """Compute distribution statistics over all non-transparent HSV pixels.

    Args:
        hsv: (N, 3) float array with H in [0, 360), S and V in [0, 1].

    Returns:
        Dict with 10 statistical features.
    """
    h = hsv[:, 0]
    s = hsv[:, 1]
    v = hsv[:, 2]

    h_rad = 2 * math.pi * h / 360.0
    mean_h_sin = float(np.mean(np.sin(h_rad)))
    mean_h_cos = float(np.mean(np.cos(h_rad)))

    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))
    std_s = float(np.std(s))
    std_v = float(np.std(v))

    # Circular std of hue (Mardia formula via scipy)
    std_hue_angular = float(circstd(h_rad))

    prop_dark = float(np.mean(v < 0.15))
    prop_saturated = float(np.mean(s > 0.7))

    # Color diversity: number of 30° hue bins with ≥ 5% of pixels
    bin_edges = np.arange(0, 361, HUE_BIN_WIDTH)
    counts, _ = np.histogram(h, bins=bin_edges)
    proportions = counts / len(h)
    color_diversity = int(np.sum(proportions >= 0.05))

    return {
        "mean_h_sin": mean_h_sin,
        "mean_h_cos": mean_h_cos,
        "mean_s": mean_s,
        "mean_v": mean_v,
        "std_s": std_s,
        "std_v": std_v,
        "std_hue_angular": std_hue_angular,
        "prop_dark": prop_dark,
        "prop_saturated": prop_saturated,
        "color_diversity": color_diversity,
    }


# ---------------------------------------------------------------------------
# Step 5: Hue histogram
# ---------------------------------------------------------------------------

def compute_hue_histogram(hsv: np.ndarray) -> dict:
    """Compute a 12-bin hue histogram (30° bins) as proportions.

    Args:
        hsv: (N, 3) float array with H in [0, 360).

    Returns:
        Dict with keys hue_bin_0, hue_bin_30, ..., hue_bin_330.
    """
    h = hsv[:, 0]
    bin_edges = np.arange(0, 361, HUE_BIN_WIDTH)
    counts, _ = np.histogram(h, bins=bin_edges)
    proportions = counts / len(h)

    features: dict = {}
    for i, prop in enumerate(proportions):
        features[f"hue_bin_{i * HUE_BIN_WIDTH}"] = float(prop)

    return features


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_features(pokemon: dict, labels: dict) -> Optional[dict]:
    """Extract all 42 color features for a single Pokémon.

    Args:
        pokemon: Metadata dict with keys: id, name, type1, type2.
        labels: Mapping from type string to integer.

    Returns:
        Feature dict ready for a DataFrame row, or None if the sprite is
        unusable.
    """
    sprite_path = RAW_DIR / f"{pokemon['id']}.png"

    if not sprite_path.exists():
        warnings.warn(f"Sprite not found: {sprite_path}")
        return None

    rgb = load_sprite_pixels(sprite_path)
    if rgb is None:
        return None

    if len(rgb) < MIN_PIXELS:
        warnings.warn(f"ID {pokemon['id']} ({pokemon['name']}): only {len(rgb)} non-transparent "
                      f"pixels, skipping.")
        return None

    hsv = rgb_to_hsv_array(rgb)

    row: dict = {
        "id": pokemon["id"],
        "name": pokemon["name"],
        "type1": pokemon["type1"],
        "type2": pokemon["type2"],
        "type1_encoded": labels.get(pokemon["type1"], -1),
    }

    row.update(compute_kmeans_features(hsv))
    row.update(compute_stat_features(hsv))
    row.update(compute_hue_histogram(hsv))

    return row


def process_all() -> None:
    """Main routine: extract features for all Pokémon and save to CSV."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    pokemon_json_path = RAW_DIR / "pokemon.json"
    with pokemon_json_path.open(encoding="utf-8") as f:
        pokemon_list: list[dict] = json.load(f)

    with LABELS_PATH.open(encoding="utf-8") as f:
        labels: dict = json.load(f)

    rows: list[dict] = []
    skipped = 0

    for idx, pokemon in enumerate(pokemon_list, start=1):
        row = extract_features(pokemon, labels)
        if row is None:
            skipped += 1
        else:
            rows.append(row)

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(pokemon_list)}...")

    df = pd.DataFrame(rows)
    output_path = PROCESSED_DIR / "features.csv"
    df.to_csv(output_path, index=False)

    print(f"\nDone. Processed: {len(rows)}, Skipped: {skipped}")
    print(f"Saved to: {output_path}")
    print("\nClass distribution (type1):")
    print(df["type1"].value_counts().to_string())


if __name__ == "__main__":
    process_all()
