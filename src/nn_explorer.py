"""
nn_explorer.py -- Tkinter GUI for interactive Neural Network explainability.

Replicates the final NN setup from notebooks/3_neural_network.ipynb:
  - Features: all 42 color features (common.FEATURE_COLS_ALL)
  - Split: common.get_train_test_split (same random seed/stratification)
  - Scaling: StandardScaler fitted on train
  - Training: SMOTE(train only) + MLPClassifier(best params from notebook)

Run from project root:
  python src/nn_explorer.py
"""

from __future__ import annotations

import bisect
import random
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Optional

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from imblearn.over_sampling import SMOTE  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

# -- Project paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import common  # noqa: E402

# -- Notebook 3 best params ---------------------------------------------------
NN_HIDDEN_LAYER_SIZES = (128, 64)
NN_ACTIVATION = "relu"
NN_ALPHA = 0.001
NN_LEARNING_RATE_INIT = 0.005
NN_MAX_ITER = 500

WINDOW_TITLE = (
    "NN Explorer - Notebook 3 Model "
    f"({len(common.FEATURE_COLS_ALL)} features, SMOTE + MLP)"
)

# -- Theme --------------------------------------------------------------------
DARK_THEME = {
    "BG": "#1e1e1e",
    "BG_LIGHT": "#2a2a2a",
    "FG": "#e0e0e0",
    "FG_DIM": "#aaaaaa",
    "GREEN": "#88cc88",
    "RED": "#ff6b6b",
    "YELLOW": "#ffcc00",
    "ACCENT": "#3a7bd5",
}

LIGHT_THEME = {
    "BG": "#f3f5f8",
    "BG_LIGHT": "#ffffff",
    "FG": "#1f2933",
    "FG_DIM": "#52606d",
    "GREEN": "#2e7d32",
    "RED": "#c62828",
    "YELLOW": "#9a6700",
    "ACCENT": "#1565c0",
}

# Mutable runtime palette used by plotting and Tk widgets.
BG = DARK_THEME["BG"]
BG_LIGHT = DARK_THEME["BG_LIGHT"]
FG = DARK_THEME["FG"]
FG_DIM = DARK_THEME["FG_DIM"]
GREEN = DARK_THEME["GREEN"]
RED = DARK_THEME["RED"]
YELLOW = DARK_THEME["YELLOW"]
ACCENT = DARK_THEME["ACCENT"]

# -- High-DPI sizing ----------------------------------------------------------
UI_SCALE = 1.45


def _sz(value: int) -> int:
    return max(8, int(round(value * UI_SCALE)))


FONT_UI_S = ("Arial", _sz(11))
FONT_UI_M = ("Arial", _sz(12))
FONT_UI_B = ("Arial", _sz(12), "bold")
FONT_TITLE = ("Arial", _sz(15), "bold")
FONT_META = ("Arial", _sz(10))
FONT_META_B = ("Arial", _sz(10), "bold")
FONT_MONO = ("Courier", _sz(9))


# ============================================================================
# Data + model
# ============================================================================

def train_nn_model() -> dict:
    """Train the notebook-aligned NN model and return artifacts for the GUI."""
    df = common.load_data()
    type_to_int, int_to_type = common.get_label_mapping()
    feature_cols = common.FEATURE_COLS_ALL

    X_train, X_test, y_train, y_test, split_idx = common.get_train_test_split(
        df,
        feature_cols=feature_cols,
    )

    scaler = common.get_scaler(X_train)
    X_train_sc = pd.DataFrame(
        scaler.transform(X_train),
        columns=feature_cols,
        index=X_train.index,
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test),
        columns=feature_cols,
        index=X_test.index,
    )

    smote = SMOTE(random_state=common.RANDOM_STATE)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_sc, y_train)

    mlp = MLPClassifier(
        hidden_layer_sizes=NN_HIDDEN_LAYER_SIZES,
        activation=NN_ACTIVATION,
        alpha=NN_ALPHA,
        learning_rate_init=NN_LEARNING_RATE_INIT,
        max_iter=NN_MAX_ITER,
        random_state=common.RANDOM_STATE,
    )
    mlp.fit(X_train_smote, y_train_smote)

    y_pred_test = mlp.predict(X_test_sc)
    y_prob_test = mlp.predict_proba(X_test_sc)

    perm_imp = permutation_importance(
        mlp,
        X_test_sc,
        y_test,
        n_repeats=8,
        random_state=common.RANDOM_STATE,
        scoring="f1_macro",
    )
    perm_series = pd.Series(perm_imp.importances_mean, index=feature_cols).sort_values(
        ascending=False,
    )

    train_meta = df.iloc[split_idx["train_idx"]].reset_index(drop=True)
    test_meta = df.iloc[split_idx["test_idx"]].reset_index(drop=True)

    all_ids = sorted(df["id"].tolist())
    test_ids = sorted(test_meta["id"].tolist())
    mis_mask = y_pred_test != y_test.values
    misclassified_ids = sorted(test_meta.loc[mis_mask, "id"].tolist())

    return {
        "df": df,
        "type_to_int": type_to_int,
        "int_to_type": int_to_type,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "mlp": mlp,
        "X_train_sc": X_train_sc,
        "X_test_sc": X_test_sc,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "y_prob_test": y_prob_test,
        "perm_series": perm_series,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "all_ids": all_ids,
        "test_ids": test_ids,
        "misclassified_ids": misclassified_ids,
    }


def _top_probabilities(
    prob_row: np.ndarray,
    classes: np.ndarray,
    int_to_type: dict[int, str],
    k: int = 5,
) -> list[tuple[str, float]]:
    """Return top-k (type_name, probability) sorted descending."""
    top_idx = np.argsort(prob_row)[::-1][:k]
    return [(int_to_type[int(classes[i])], float(prob_row[i])) for i in top_idx]


def _hidden_activation_fn(name: str, z: np.ndarray) -> np.ndarray:
    """Apply hidden-layer activation used by sklearn MLPClassifier."""
    if name == "relu":
        return np.maximum(0.0, z)
    if name == "tanh":
        return np.tanh(z)
    if name == "logistic":
        return 1.0 / (1.0 + np.exp(-z))
    if name == "identity":
        return z
    return z


def _forward_hidden_activations(
    mlp: MLPClassifier,
    x_scaled: np.ndarray,
) -> list[np.ndarray]:
    """Return hidden-layer activations for a single scaled sample."""
    a = x_scaled.reshape(1, -1)
    hidden_acts: list[np.ndarray] = []

    for w, b in zip(mlp.coefs_[:-1], mlp.intercepts_[:-1]):
        z = a @ w + b
        a = _hidden_activation_fn(mlp.activation, z)
        hidden_acts.append(a.ravel())

    return hidden_acts


def _input_to_neuron_impact(
    mlp: MLPClassifier,
    x_scaled: np.ndarray,
    pred_class_int: int,
    feature_cols: list[str],
    hidden_acts: list[np.ndarray],
) -> tuple[list[tuple[str, float]], list[tuple[int, float]]]:
    """Compute feature->neuron impact proxy for the predicted class."""
    classes = np.array(mlp.classes_)
    pred_idx = int(np.where(classes == pred_class_int)[0][0])

    strength = np.abs(mlp.coefs_[-1][:, pred_idx])
    for w in reversed(mlp.coefs_[1:-1]):
        strength = np.abs(w) @ strength

    h1 = hidden_acts[0] if hidden_acts else np.array([])
    n_h1 = len(h1)
    if n_h1 == 0:
        return [], []

    neuron_scores = np.abs(h1) * strength
    top_neuron_idx = np.argsort(neuron_scores)[::-1][:12]
    top_neurons = [(int(i), float(neuron_scores[i])) for i in top_neuron_idx]

    w0 = mlp.coefs_[0]
    per_feat = np.sum(np.abs(x_scaled.reshape(-1, 1) * w0) * neuron_scores.reshape(1, -1), axis=1)
    top_feat_idx = np.argsort(per_feat)[::-1][:12]
    top_features = [(feature_cols[i], float(per_feat[i])) for i in top_feat_idx]

    return top_features, top_neurons


def query_pokemon(m: dict, pokemon_id: int) -> Optional[dict]:
    """Look up one Pokemon and compute explanation panels data."""
    df = m["df"]
    matches = df[df["id"] == pokemon_id]
    if matches.empty:
        return None

    row = matches.iloc[0]
    feature_cols = m["feature_cols"]

    x_raw = pd.DataFrame(
        [row[feature_cols].values.astype(float)],
        columns=feature_cols,
    )
    x_sc = m["scaler"].transform(x_raw)[0]
    x_sc_df = pd.DataFrame([x_sc], columns=feature_cols)

    mlp = m["mlp"]
    pred_int = int(mlp.predict(x_sc_df)[0])
    prob_row = mlp.predict_proba(x_sc_df)[0]
    pred_label = m["int_to_type"][pred_int]
    true_label = row["type1"]

    in_test = pokemon_id in m["test_ids"]
    top_probs = _top_probabilities(prob_row, mlp.classes_, m["int_to_type"], k=5)

    # Nearest prototype/counterfactual over real training Pokemon only.
    X_train_arr = m["X_train_sc"].to_numpy()
    y_train_arr = m["y_train"].to_numpy()

    same_pos = np.where(y_train_arr == pred_int)[0]
    if same_pos.size == 0:
        same_pos = np.arange(len(y_train_arr))
    same_dists = np.linalg.norm(X_train_arr[same_pos] - x_sc, axis=1)
    best_same = int(np.argmin(same_dists))
    proto_pos = int(same_pos[best_same])
    proto_meta = m["train_meta"].iloc[proto_pos]

    other_pos = np.where(y_train_arr != pred_int)[0]
    if other_pos.size == 0:
        other_pos = np.arange(len(y_train_arr))
    other_dists = np.linalg.norm(X_train_arr[other_pos] - x_sc, axis=1)
    best_other = int(np.argmin(other_dists))
    cf_pos = int(other_pos[best_other])
    cf_meta = m["train_meta"].iloc[cf_pos]

    hidden_acts = _forward_hidden_activations(mlp, x_sc)
    feat_neuron_top12, neuron_top12 = _input_to_neuron_impact(
        mlp,
        x_sc,
        pred_class_int=pred_int,
        feature_cols=feature_cols,
        hidden_acts=hidden_acts,
    )

    return {
        "pokemon_id": pokemon_id,
        "name": row["name"],
        "true_type": true_label,
        "type2": row["type2"] if pd.notna(row["type2"]) else None,
        "pred_type": pred_label,
        "correct": pred_label == true_label,
        "in_test": in_test,
        "top_probs": top_probs,
        "perm_top15": list(m["perm_series"].head(15).items()),
        "hidden_activations": hidden_acts,
        "feature_neuron_top12": feat_neuron_top12,
        "neuron_top12": neuron_top12,
        "prototype": {
            "name": proto_meta["name"],
            "id": int(proto_meta["id"]),
            "type1": proto_meta["type1"],
            "distance": float(same_dists[best_same]),
        },
        "counterfactual": {
            "name": cf_meta["name"],
            "id": int(cf_meta["id"]),
            "type1": cf_meta["type1"],
            "distance": float(other_dists[best_other]),
        },
        "raw_features": {f: float(row[f]) for f in feature_cols},
    }


# ============================================================================
# Figure builders
# ============================================================================

def _load_sprite(pokemon_id: int, size: tuple[int, int] | None = None) -> Image.Image | None:
    """Load a sprite composited on dark background, optionally resized."""
    path = common.get_sprite_path(pokemon_id)
    if not path.exists():
        return None
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (30, 30, 30, 255))
    bg.paste(img, mask=img.split()[3])
    if size is not None:
        bg = bg.resize(size, Image.NEAREST)
    return bg


def build_probabilities_chart(top_probs: list[tuple[str, float]]) -> plt.Figure:
    """Bar chart of top-5 predicted probabilities."""
    labels = [t for t, _ in top_probs][::-1]
    values = [p * 100.0 for _, p in top_probs][::-1]
    colors = [common.TYPE_COLORS.get(t, "#777777") for t in labels]

    fig, ax = plt.subplots(figsize=(5.8, 3.0))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG_LIGHT)

    bars = ax.barh(labels, values, color=colors, edgecolor=BG, height=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=_sz(9), color=FG)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", fontsize=_sz(10), color=FG_DIM)
    ax.set_title("Prediction Confidence (Top 5)", fontsize=_sz(11), color=FG)
    ax.tick_params(axis="both", labelsize=_sz(9), colors=FG)
    for spine in ax.spines.values():
        spine.set_color("#444444")

    fig.tight_layout(pad=0.5)
    return fig


def build_global_importance_chart(perm_top15: list[tuple[str, float]]) -> plt.Figure:
    """Permutation importance chart (global, from test set)."""
    names = [n for n, _ in perm_top15][::-1]
    vals = [float(v) for _, v in perm_top15][::-1]

    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG_LIGHT)

    max_v = max(vals) if vals else 1.0
    colors = ["#cc7755" if v >= max_v * 0.65 else "#88aadd" for v in vals]
    ax.barh(names, vals, color=colors, edgecolor=BG, height=0.7)

    ax.set_title("Global Importance (Permutation, Top 15)", fontsize=_sz(11), color=FG)
    ax.set_xlabel("Mean decrease in f1_macro", fontsize=_sz(10), color=FG_DIM)
    ax.tick_params(axis="both", labelsize=_sz(9), colors=FG_DIM)
    for spine in ax.spines.values():
        spine.set_color("#444444")

    fig.tight_layout(pad=0.5)
    return fig


def build_activation_heatmap(hidden_activations: list[np.ndarray]) -> plt.Figure:
    """Heatmap of hidden-layer activations for the current sample."""
    n_layers = len(hidden_activations)
    max_units = max((len(a) for a in hidden_activations), default=1)
    mat = np.full((n_layers, max_units), np.nan)

    for i, acts in enumerate(hidden_activations):
        mat[i, : len(acts)] = acts

    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG_LIGHT)

    im = ax.imshow(mat, aspect="auto", cmap="coolwarm", interpolation="nearest")
    ax.set_yticks(np.arange(n_layers))
    ax.set_yticklabels([f"Hidden {i + 1}" for i in range(n_layers)], color=FG, fontsize=_sz(9))
    ax.set_xlabel("Neuron index", fontsize=_sz(10), color=FG_DIM)
    ax.set_title("Neuron Activations by Layer", fontsize=_sz(11), color=FG)
    ax.tick_params(axis="x", labelsize=_sz(9), colors=FG_DIM)
    for spine in ax.spines.values():
        spine.set_color("#444444")

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.ax.tick_params(labelsize=_sz(8), colors=FG_DIM)
    cbar.outline.set_edgecolor("#444444")
    cbar.set_label("Activation", color=FG_DIM, fontsize=_sz(9))

    fig.tight_layout(pad=0.5)
    return fig


def build_feature_neuron_impact_chart(feature_top12: list[tuple[str, float]]) -> plt.Figure:
    """Bar chart of features with strongest input->neuron impact."""
    names = [n for n, _ in feature_top12][::-1]
    vals = [v for _, v in feature_top12][::-1]

    fig, ax = plt.subplots(figsize=(6.3, 4.8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG_LIGHT)

    max_v = max(vals) if vals else 1.0
    colors = ["#ff9966" if v >= max_v * 0.7 else "#77aadd" for v in vals]
    ax.barh(names, vals, color=colors, edgecolor=BG, height=0.65)
    ax.set_title("Feature Impact Through Neurons (Top 12)", fontsize=_sz(11), color=FG)
    ax.set_xlabel("Aggregated impact score", fontsize=_sz(10), color=FG_DIM)
    ax.tick_params(axis="both", labelsize=_sz(9), colors=FG_DIM)
    for spine in ax.spines.values():
        spine.set_color("#444444")

    fig.tight_layout(pad=0.5)
    return fig


def build_proto_counterfactual_panel(proto: dict, cf: dict, pred_type: str) -> plt.Figure:
    """Panel with nearest predicted-type prototype and nearest other-type sample."""
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    fig.patch.set_facecolor(BG)

    left, right = axes
    for ax in axes:
        ax.set_facecolor(BG)
        ax.axis("off")

    p_sprite = _load_sprite(proto["id"])
    if p_sprite is not None:
        left.imshow(p_sprite)
    p_color = common.TYPE_COLORS.get(proto["type1"], FG)
    left.set_title(
        "Nearest Prototype\n"
        f"{proto['name'].capitalize()} ({proto['type1']})\n"
        f"dist={proto['distance']:.3f}",
        fontsize=_sz(9),
        color=p_color,
    )

    c_sprite = _load_sprite(cf["id"])
    if c_sprite is not None:
        right.imshow(c_sprite)
    c_color = common.TYPE_COLORS.get(cf["type1"], FG)
    right.set_title(
        "Nearest Counterfactual\n"
        f"{cf['name'].capitalize()} ({cf['type1']})\n"
        f"dist={cf['distance']:.3f}",
        fontsize=_sz(9),
        color=c_color,
    )

    fig.suptitle(
        f"Prototype vs Counterfactual around predicted class '{pred_type}'",
        fontsize=_sz(10),
        color=FG_DIM,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def build_color_swatches(features: dict[str, float]) -> plt.Figure:
    """Dominant-color strip from dom1..dom5 proportions."""
    fig, ax = plt.subplots(figsize=(2.8, 0.7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x = 0.0
    for i in range(1, 6):
        h = features.get(f"dom{i}_h", 0.0) / 360.0
        s = features.get(f"dom{i}_s", 0.0)
        v = features.get(f"dom{i}_v", 0.0)
        prop = features.get(f"dom{i}_prop", 0.0)
        color = np.clip(mcolors.hsv_to_rgb([h, s, v]), 0.0, 1.0)
        ax.bar(x + prop / 2.0, 1.0, width=prop, color=color, edgecolor=BG, linewidth=0.5)
        x += prop

    ax.set_xlim(0, max(x, 0.01))
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Dominant Colors", color=FG_DIM, fontsize=_sz(8), pad=2)
    fig.tight_layout(pad=0.2)
    return fig


# ============================================================================
# GUI
# ============================================================================

class NNExplorer:
    """Interactive NN explainability GUI aligned with notebook 3."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1780x1080")
        self.root.minsize(1400, 820)
        self.root.configure(bg=BG)
        self.theme_name = "dark"

        self.model: Optional[dict] = None
        self.current_id: Optional[int] = None
        self._canvas_refs: list[FigureCanvasTkAgg] = []
        self._themed_buttons: list[tk.Button] = []
        self._label_widgets: list[tk.Label] = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._load_model)

    def _build_ui(self) -> None:
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill=tk.X, padx=10, pady=6)
        self.top_bar = top

        lbl_id = tk.Label(top, text="Pokemon ID:", bg=BG, fg=FG, font=FONT_UI_M)
        lbl_id.pack(side=tk.LEFT)
        self._label_widgets.append(lbl_id)

        self.id_entry = tk.Entry(top, width=7, font=FONT_UI_M)
        self.id_entry.pack(side=tk.LEFT, padx=(4, 8))
        self.id_entry.bind("<Return>", lambda _: self._explain_from_entry())

        btn_explain = tk.Button(
            top,
            text="Explain",
            command=self._explain_from_entry,
            bg=ACCENT,
            fg="white",
            font=FONT_UI_B,
            relief=tk.FLAT,
            padx=8,
        )
        btn_explain.pack(side=tk.LEFT, padx=(0, 12))
        self._themed_buttons.append(btn_explain)

        btn_prev = tk.Button(
            top,
            text="<",
            command=self._prev,
            bg="#444",
            fg=FG,
            font=FONT_UI_M,
            relief=tk.FLAT,
            width=3,
        )
        btn_prev.pack(side=tk.LEFT, padx=2)
        self._themed_buttons.append(btn_prev)

        btn_next = tk.Button(
            top,
            text=">",
            command=self._next,
            bg="#444",
            fg=FG,
            font=FONT_UI_M,
            relief=tk.FLAT,
            width=3,
        )
        btn_next.pack(side=tk.LEFT, padx=(2, 10))
        self._themed_buttons.append(btn_next)

        btn_random = tk.Button(
            top,
            text="Random",
            command=self._random,
            bg="#555",
            fg=FG,
            font=FONT_UI_M,
            relief=tk.FLAT,
            padx=6,
        )
        btn_random.pack(side=tk.LEFT, padx=2)
        self._themed_buttons.append(btn_random)

        btn_mis = tk.Button(
            top,
            text="Random Misclassified",
            command=self._random_misclassified,
            bg="#8B0000",
            fg="white",
            font=FONT_UI_M,
            relief=tk.FLAT,
            padx=6,
        )
        btn_mis.pack(side=tk.LEFT, padx=(2, 10))
        self._themed_buttons.append(btn_mis)

        self.mode_var = tk.StringVar(value="Test set only")
        mode_combo = ttk.Combobox(
            top,
            textvariable=self.mode_var,
            values=["Test set only", "All Pokemon"],
            state="readonly",
            width=14,
            font=FONT_UI_S,
            style="Theme.TCombobox",
        )
        mode_combo.pack(side=tk.LEFT, padx=(0, 12))
        self.mode_combo = mode_combo

        self.light_theme_var = tk.BooleanVar(value=False)
        self.theme_toggle = tk.Checkbutton(
            top,
            text="Light Theme",
            variable=self.light_theme_var,
            command=self._toggle_theme,
            bg=BG,
            fg=FG,
            selectcolor=BG_LIGHT,
            activebackground=BG,
            activeforeground=FG,
            font=FONT_UI_S,
        )
        self.theme_toggle.pack(side=tk.LEFT, padx=(0, 12))

        self.status_label = tk.Label(
            top,
            text="Loading model...",
            bg=BG,
            fg=FG_DIM,
            font=FONT_UI_S,
        )
        self.status_label.pack(side=tk.LEFT, padx=8)

        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))
        self.main_frame = main

        self.left_frame = tk.Frame(main, bg=BG, width=280)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        self.left_frame.pack_propagate(False)

        self.center_frame = tk.Frame(main, bg=BG)
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.center_frame.pack_propagate(False)

        self.right_frame = tk.Frame(main, bg=BG)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right_frame.pack_propagate(False)

        # Split center and right into top/bottom regions so the lower half is used.
        self.center_top = tk.Frame(self.center_frame, bg=BG)
        self.center_top.pack(fill=tk.BOTH, expand=True, pady=(0, 4))
        self.center_top.pack_propagate(False)
        self.center_bottom = tk.Frame(self.center_frame, bg=BG)
        self.center_bottom.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.center_bottom.pack_propagate(False)

        self.right_top = tk.Frame(self.right_frame, bg=BG)
        self.right_top.pack(fill=tk.BOTH, expand=True, pady=(0, 4))
        self.right_top.pack_propagate(False)
        self.right_bottom = tk.Frame(self.right_frame, bg=BG)
        self.right_bottom.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.right_bottom.pack_propagate(False)

        self._render_frames = [
            self.left_frame,
            self.center_top,
            self.center_bottom,
            self.right_top,
            self.right_bottom,
        ]

        self._apply_theme_widgets()

        self.root.bind("<Left>", lambda _: self._prev())
        self.root.bind("<Right>", lambda _: self._next())
        self.root.bind("<r>", lambda _: self._random())
        self.root.bind("<m>", lambda _: self._random_misclassified())

    def _load_model(self) -> None:
        try:
            self.model = train_nn_model()
            n = len(self.model["df"])
            nt = len(self.model["test_ids"])
            nm = len(self.model["misclassified_ids"])
            self.status_label.config(
                text=f"Ready - {n} Pokemon | {nt} test | {nm} misclassified",
                fg=GREEN,
            )
        except Exception as exc:
            self.status_label.config(text=f"Error: {exc}", fg=RED)

    def _set_theme_palette(self, theme_name: str) -> None:
        """Update module-level palette used by Tk widgets and matplotlib plots."""
        global BG, BG_LIGHT, FG, FG_DIM, GREEN, RED, YELLOW, ACCENT

        palette = LIGHT_THEME if theme_name == "light" else DARK_THEME
        BG = palette["BG"]
        BG_LIGHT = palette["BG_LIGHT"]
        FG = palette["FG"]
        FG_DIM = palette["FG_DIM"]
        GREEN = palette["GREEN"]
        RED = palette["RED"]
        YELLOW = palette["YELLOW"]
        ACCENT = palette["ACCENT"]

    def _apply_theme_widgets(self) -> None:
        """Apply current palette to existing Tk widgets and ttk styles."""
        self.root.configure(bg=BG)
        self.top_bar.configure(bg=BG)
        self.main_frame.configure(bg=BG)

        for frame in [
            self.left_frame,
            self.center_frame,
            self.right_frame,
            self.center_top,
            self.center_bottom,
            self.right_top,
            self.right_bottom,
        ]:
            frame.configure(bg=BG)

        for lbl in self._label_widgets:
            lbl.configure(bg=BG, fg=FG)

        # Re-apply role-based button colors.
        if len(self._themed_buttons) >= 5:
            self._themed_buttons[0].configure(bg=ACCENT, fg="white", activebackground=ACCENT)
            self._themed_buttons[1].configure(bg="#666666", fg=FG, activebackground="#666666")
            self._themed_buttons[2].configure(bg="#666666", fg=FG, activebackground="#666666")
            self._themed_buttons[3].configure(bg="#7a7a7a", fg=FG, activebackground="#7a7a7a")
            self._themed_buttons[4].configure(bg="#8B0000", fg="white", activebackground="#8B0000")

        self.id_entry.configure(bg=BG_LIGHT, fg=FG, insertbackground=FG, relief=tk.FLAT)
        self.status_label.configure(bg=BG, fg=FG_DIM)
        self.theme_toggle.configure(
            bg=BG,
            fg=FG,
            selectcolor=BG_LIGHT,
            activebackground=BG,
            activeforeground=FG,
        )

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Theme.TCombobox",
            fieldbackground=BG_LIGHT,
            background=BG,
            foreground=FG,
            arrowcolor=FG,
            bordercolor="#666666",
        )
        style.map(
            "Theme.TCombobox",
            fieldbackground=[("readonly", BG_LIGHT)],
            foreground=[("readonly", FG)],
            background=[("readonly", BG)],
        )

    def _toggle_theme(self) -> None:
        self.theme_name = "light" if self.light_theme_var.get() else "dark"
        self._set_theme_palette(self.theme_name)
        self._apply_theme_widgets()

        if self.current_id is not None and self.model is not None:
            self._explain(self.current_id)

    def _on_close(self) -> None:
        plt.close("all")
        self.root.destroy()

    def _id_list(self) -> list[int]:
        if self.model is None:
            return []
        if self.mode_var.get() == "Test set only":
            return self.model["test_ids"]
        return self.model["all_ids"]

    def _navigate(self, direction: int) -> None:
        ids = self._id_list()
        if not ids:
            return

        if self.current_id is None:
            self._explain(ids[0] if direction == 1 else ids[-1])
            return

        pos = bisect.bisect_left(ids, self.current_id)
        if pos < len(ids) and ids[pos] == self.current_id:
            new_pos = (pos + direction) % len(ids)
        else:
            new_pos = min(pos, len(ids) - 1) if direction == 1 else max(pos - 1, 0)
        self._explain(ids[new_pos])

    def _prev(self) -> None:
        self._navigate(-1)

    def _next(self) -> None:
        self._navigate(1)

    def _random(self) -> None:
        ids = self._id_list()
        if ids:
            self._explain(random.choice(ids))

    def _random_misclassified(self) -> None:
        if self.model and self.model["misclassified_ids"]:
            self._explain(random.choice(self.model["misclassified_ids"]))

    def _explain_from_entry(self) -> None:
        raw = self.id_entry.get().strip()
        if not raw.isdigit():
            self.status_label.config(text="Enter a numeric Pokedex ID.", fg=RED)
            return
        self._explain(int(raw))

    def _explain(self, pokemon_id: int) -> None:
        if self.model is None:
            self.status_label.config(text="Model not loaded yet.", fg=RED)
            return

        # Keep the current toplevel size stable across re-renders.
        cur_w = self.root.winfo_width()
        cur_h = self.root.winfo_height()

        result = query_pokemon(self.model, pokemon_id)
        if result is None:
            self.status_label.config(text=f"ID {pokemon_id} not found.", fg=RED)
            return

        self.current_id = pokemon_id
        self.id_entry.delete(0, tk.END)
        self.id_entry.insert(0, str(pokemon_id))

        self._clear_panels()
        self._render(result)

        if cur_w > 100 and cur_h > 100:
            self.root.geometry(f"{cur_w}x{cur_h}")

        tag = "test" if result["in_test"] else "train"
        sym = "OK" if result["correct"] else "MISS"
        self.status_label.config(
            text=(
                f"#{pokemon_id} {result['name'].capitalize()} [{tag}] "
                f"{result['true_type']}->{result['pred_type']} {sym}"
            ),
            fg=GREEN if result["correct"] else RED,
        )

    def _clear_panels(self) -> None:
        for frame in self._render_frames:
            for w in frame.winfo_children():
                w.destroy()
        for c in self._canvas_refs:
            plt.close(c.figure)
        self._canvas_refs.clear()
        plt.close("all")

    def _embed(self, fig: plt.Figure, parent: tk.Frame, **pack_kw) -> None:
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(**pack_kw)
        self._canvas_refs.append(canvas)

    def _render(self, r: dict) -> None:
        self._render_header(r)
        self._render_center(r)
        self._render_right(r)

    def _render_header(self, r: dict) -> None:
        f = self.left_frame

        sprite = _load_sprite(r["pokemon_id"], size=(170, 170))
        if sprite is not None:
            photo = ImageTk.PhotoImage(sprite)
            lbl = tk.Label(f, image=photo, bg=BG)
            lbl.image = photo
            lbl.pack(pady=(10, 4))

        tk.Label(
            f,
            text=f"{r['name'].capitalize()}  #{r['pokemon_id']}",
            bg=BG,
            fg=FG,
            font=FONT_TITLE,
        ).pack(pady=(2, 4))

        true_c = common.TYPE_COLORS.get(r["true_type"], FG)
        tk.Label(
            f,
            text=f"True: {r['true_type']}",
            bg=BG,
            fg=true_c,
            font=FONT_UI_M,
        ).pack()

        pred_c = common.TYPE_COLORS.get(r["pred_type"], FG)
        sym = "  OK" if r["correct"] else "  MISS"
        sym_c = GREEN if r["correct"] else RED
        pf = tk.Frame(f, bg=BG)
        pf.pack(pady=2)
        tk.Label(pf, text=f"Pred: {r['pred_type']}", bg=BG, fg=pred_c, font=FONT_UI_M).pack(
            side=tk.LEFT,
        )
        tk.Label(pf, text=sym, bg=BG, fg=sym_c, font=FONT_UI_B).pack(side=tk.LEFT)

        t2 = r["type2"] if r["type2"] else "-"
        tag = "test set" if r["in_test"] else "train set"
        tk.Label(
            f,
            text=f"Type 2: {t2}  |  {tag}",
            bg=BG,
            fg=FG_DIM,
            font=FONT_META,
        ).pack(pady=2)

        if r["type2"] and r["pred_type"] == r["type2"]:
            tk.Label(
                f,
                text="* Pred matches type2",
                bg=BG,
                fg=YELLOW,
                font=FONT_META_B,
            ).pack()

        max_prob = r["top_probs"][0][1] if r["top_probs"] else 0.0
        tk.Label(
            f,
            text=f"Confidence: {max_prob * 100.0:.1f}%",
            bg=BG,
            fg=FG,
            font=FONT_UI_M,
        ).pack(pady=(4, 2))

        self._embed(build_color_swatches(r["raw_features"]), f, fill=tk.X, padx=4, pady=2)

    def _render_center(self, r: dict) -> None:
        self._embed(
            build_probabilities_chart(r["top_probs"]),
            self.center_top,
            fill=tk.X,
            padx=2,
            pady=(0, 6),
        )
        self._embed(
            build_activation_heatmap(r["hidden_activations"]),
            self.center_top,
            fill=tk.BOTH,
            expand=True,
            padx=2,
            pady=(0, 2),
        )
        self._embed(
            build_proto_counterfactual_panel(r["prototype"], r["counterfactual"], r["pred_type"]),
            self.center_bottom,
            fill=tk.BOTH,
            expand=True,
            padx=2,
            pady=2,
        )

    def _render_right(self, r: dict) -> None:
        self._embed(
            build_global_importance_chart(r["perm_top15"]),
            self.right_top,
            fill=tk.BOTH,
            expand=True,
            padx=2,
            pady=2,
        )

        self._embed(
            build_feature_neuron_impact_chart(r["feature_neuron_top12"]),
            self.right_bottom,
            fill=tk.BOTH,
            expand=True,
            padx=2,
            pady=(0, 6),
        )

        tk.Label(
            self.right_bottom,
            text="Top activated neurons",
            bg=BG,
            fg=FG_DIM,
            font=FONT_META_B,
        ).pack(anchor=tk.W, padx=4)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Dark.Treeview",
            background=BG_LIGHT,
            foreground=FG,
            fieldbackground=BG_LIGHT,
            font=FONT_MONO,
            rowheight=_sz(22),
        )
        style.configure(
            "Dark.Treeview.Heading",
            background="#444",
            foreground=FG,
            font=FONT_META_B,
        )

        tf = tk.Frame(self.right_bottom, bg=BG)
        tf.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        tree = ttk.Treeview(
            tf,
            columns=("neuron", "score"),
            show="headings",
            style="Dark.Treeview",
            height=10,
        )
        tree.heading("neuron", text="Neuron (Hidden 1)")
        tree.heading("score", text="Act. Score")
        tree.column("neuron", width=180, anchor=tk.W)
        tree.column("score", width=110, anchor=tk.E)

        sb = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        for idx, score in r["neuron_top12"]:
            tree.insert("", tk.END, values=(f"h1_{idx}", f"{score:.4f}"))


def main() -> None:
    """Launch the NN explorer app."""
    root = tk.Tk()
    root.tk.call("tk", "scaling", UI_SCALE)
    NNExplorer(root)
    root.mainloop()


if __name__ == "__main__":
    main()