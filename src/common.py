"""
common.py — Shared utilities for all XAI notebooks.

Centralizes data loading, splitting, scaling, and visualization helpers
so all notebooks use identical data partitions and styling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.utils.class_weight import compute_class_weight

# ─────────────────────────── Constants ────────────────────────────────────────

RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLS_TIER1 = [
    "dom1_h", "dom1_s", "dom1_v", "dom1_prop",
    "dom2_h", "dom2_s", "dom2_v", "dom2_prop",
    "dom3_h", "dom3_s", "dom3_v", "dom3_prop",
    "dom4_h", "dom4_s", "dom4_v", "dom4_prop",
    "dom5_h", "dom5_s", "dom5_v", "dom5_prop",
]

FEATURE_COLS_TIER2 = [
    "mean_h_sin", "mean_h_cos", "mean_s", "mean_v",
    "std_s", "std_v", "std_hue_angular",
    "prop_dark", "prop_saturated", "color_diversity",
]

FEATURE_COLS_TIER3 = [
    "hue_bin_0", "hue_bin_30", "hue_bin_60", "hue_bin_90",
    "hue_bin_120", "hue_bin_150", "hue_bin_180", "hue_bin_210",
    "hue_bin_240", "hue_bin_270", "hue_bin_300", "hue_bin_330",
]

FEATURE_COLS_ALL = FEATURE_COLS_TIER1 + FEATURE_COLS_TIER2 + FEATURE_COLS_TIER3

META_COLS = ["id", "name", "type1", "type2", "type1_encoded"]

# Canonical color for each Pokémon type (Bulbapedia palette)
TYPE_COLORS = {
    "bug":      "#A8B820",
    "dark":     "#705848",
    "dragon":   "#7038F8",
    "electric": "#F8D030",
    "fairy":    "#EE99AC",
    "fighting": "#C03028",
    "fire":     "#F08030",
    "flying":   "#A890F0",
    "ghost":    "#705898",
    "grass":    "#78C850",
    "ground":   "#E0C068",
    "ice":      "#98D8D8",
    "normal":   "#A8A878",
    "poison":   "#A040A0",
    "psychic":  "#F85888",
    "rock":     "#B8A038",
    "steel":    "#B8B8D0",
    "water":    "#6890F0",
}

# Chosen after running 1_decision_tree.ipynb and inspecting misclassification candidates.
# correct_clear:      Gyarados  (#130, water/flying)  — obviously blue, correctly water
# correct_surprise:   Iron Crown (#1023, steel/psychic) — metallic legend, correctly steel
# misclass_secondary: Gengar    (#94,  ghost/poison)  — purple body → predicted poison
# misclass_wrong:     Gholdengo (#1000, steel/ghost)  — golden color → predicted electric
EXPLAIN_IDS: dict[str, Optional[int]] = {
    "correct_clear":      130,
    "correct_surprise":   1023,
    "misclass_secondary": 94,
    "misclass_wrong":     1000,
}


# ─────────────────────────── Project root ─────────────────────────────────────

def _find_project_root() -> Path:
    """Walk up from this file until a directory containing 'data/' is found."""
    candidate = Path(__file__).resolve().parent
    for _ in range(6):
        if (candidate / "data").exists():
            return candidate
        candidate = candidate.parent
    raise FileNotFoundError("Cannot locate project root (no 'data/' directory found).")


PROJECT_ROOT = _find_project_root()


# ─────────────────────────── Data loading ─────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load features.csv and apply Flying-type reclassification.

    Flying is not a useful primary-type class (only 9 samples, mostly
    dual-types where flying is incidental). Rules applied:
      - Pure Flying (type2 is NaN)  → type1 = 'normal'
      - Dual-type Flying            → swap type1 and type2

    Returns:
        DataFrame with all 42 features, metadata, and corrected labels.
    """
    csv_path = PROJECT_ROOT / "data" / "processed" / "features.csv"
    df = pd.read_csv(csv_path)

    _, int_to_type = get_label_mapping()
    type_to_int = {v: k for k, v in int_to_type.items()}

    flying_mask = df["type1"] == "flying"
    pure_flying  = flying_mask & df["type2"].isna()
    dual_flying  = flying_mask & df["type2"].notna()

    df.loc[pure_flying, "type1"] = "normal"
    # swap type1 <-> type2 for dual-flying
    tmp = df.loc[dual_flying, "type1"].copy()
    df.loc[dual_flying, "type1"] = df.loc[dual_flying, "type2"]
    df.loc[dual_flying, "type2"] = tmp

    df["type1_encoded"] = df["type1"].map(type_to_int)
    df["id"] = df["id"].astype(int)
    return df


def get_label_mapping() -> tuple[dict[str, int], dict[int, str]]:
    """Load labels.json and return both direction mappings.

    Returns:
        (type_to_int, int_to_type) — both mapping directions.
    """
    labels_path = PROJECT_ROOT / "data" / "labels.json"
    with labels_path.open() as f:
        type_to_int: dict[str, int] = json.load(f)
    int_to_type = {v: k for k, v in type_to_int.items()}
    return type_to_int, int_to_type


# ─────────────────────────── Train / test split ────────────────────────────────

def get_train_test_split(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, dict]:
    """Stratified train/test split, fixed seed for reproducibility.

    Args:
        df: Full feature DataFrame from load_data().
        feature_cols: Feature columns to use. Defaults to FEATURE_COLS_ALL.

    Returns:
        (X_train, X_test, y_train, y_test, split_indices)
        split_indices = {"train_idx": ndarray, "test_idx": ndarray}
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLS_ALL

    X = df[feature_cols]
    y = df["type1_encoded"]
    idx = np.arange(len(df))

    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, idx,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X_train, X_test, y_train, y_test, {"train_idx": train_idx, "test_idx": test_idx}


# ─────────────────────────── Scaling ──────────────────────────────────────────

def get_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit and return a StandardScaler on X_train.

    Apply .transform() separately to X_train and X_test.

    Args:
        X_train: Training features.

    Returns:
        Fitted StandardScaler.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


# ─────────────────────────── Class weights ────────────────────────────────────

def get_class_weights(y_train: pd.Series) -> dict[int, float]:
    """Compute balanced class weights to handle imbalance.

    Args:
        y_train: Training labels (integer-encoded).

    Returns:
        Dict mapping class label -> weight.
    """
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def get_sample_weights(y_train: pd.Series) -> np.ndarray:
    """Per-sample weights derived from balanced class weights.

    Useful for models that accept sample_weight but not class_weight.

    Args:
        y_train: Training labels.

    Returns:
        1-D array of per-sample weights.
    """
    cw = get_class_weights(y_train)
    return np.array([cw[int(label)] for label in y_train])


# ─────────────────────────── Cross-validation ─────────────────────────────────

def get_cv_splitter(n_splits: int = 5) -> StratifiedKFold:
    """Return a reproducible StratifiedKFold splitter.

    Args:
        n_splits: Number of folds.

    Returns:
        StratifiedKFold instance.
    """
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


# ─────────────────────────── Path helpers ─────────────────────────────────────

def get_sprite_path(pokemon_id: int) -> Path:
    """Return the path to a Pokemon's front sprite PNG.

    Args:
        pokemon_id: Pokedex number.

    Returns:
        Path to data/raw/{pokemon_id}.png
    """
    return PROJECT_ROOT / "data" / "raw" / f"{pokemon_id}.png"


# ─────────────────────────── Visualization helpers ────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    int_to_type: dict[int, str],
    title: str = "",
    ax: Optional[plt.Axes] = None,
    normalize: str = "true",
) -> None:
    """Normalized confusion matrix with type names as labels.

    Diagonal cells (correct predictions) are highlighted in orange so the
    eye is immediately drawn to per-class recall without searching.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        int_to_type: Mapping from integer to type name.
        title: Optional plot title.
        ax: Axes to draw on (creates new figure if None).
        normalize: 'true' normalizes by row (recall per class).
    """
    import matplotlib.patches as mpatches

    labels = sorted(int_to_type.keys())
    display_labels = [int_to_type[i] for i in labels]
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        labels=labels,
        display_labels=display_labels,
        normalize=normalize,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    target_ax = disp.ax_
    n = len(labels)
    for i in range(n):
        target_ax.add_patch(
            mpatches.Rectangle(
                (i - 0.5, i - 0.5), 1, 1,
                linewidth=2.5, edgecolor="#e07b00", facecolor="none", zorder=3,
            )
        )
    if title:
        target_ax.set_title(title, fontsize=12)
    plt.tight_layout()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    int_to_type: dict[int, str],
) -> None:
    """Print classification report with Pokemon type names.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        int_to_type: Mapping from integer to type name.
    """
    labels = sorted(int_to_type.keys())
    target_names = [int_to_type[i] for i in labels]
    print(classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    ))
