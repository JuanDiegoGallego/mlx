"""
knn_explorer.py  --  Tkinter GUI for interactive KNN explainability (Model D).

Replicates Model D from notebooks/2_knn_final.ipynb:
  Pipeline: StandardScaler -> SelectKBest(f_classif, k=15)
            -> KNeighborsClassifier(n_neighbors=5, weights='distance',
                                     metric='manhattan')
  Features: CORE_FEATURES (27 columns), no SMOTE.

Run from project root:  python src/knn_explorer.py
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from PIL import Image, ImageTk  # noqa: E402
from sklearn.feature_selection import SelectKBest, f_classif  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src import common  # noqa: E402

# ── Model D hyperparameters (from notebook GridSearchCV) ─────────────────────
KNN_N_NEIGHBORS = 5
KNN_WEIGHTS = "distance"
KNN_METRIC = "manhattan"
SELECTKBEST_K = 15

CORE_FEATURES: list[str] = [
    "dom1_h", "dom1_s", "dom1_v", "dom1_prop",
    "dom2_h", "dom2_s", "dom2_v", "dom2_prop",
    "mean_h_sin", "mean_h_cos", "mean_s", "mean_v",
    "prop_dark", "prop_saturated", "color_diversity",
    "hue_bin_0", "hue_bin_30", "hue_bin_60", "hue_bin_90",
    "hue_bin_120", "hue_bin_150", "hue_bin_180", "hue_bin_210",
    "hue_bin_240", "hue_bin_270", "hue_bin_300", "hue_bin_330",
]

WINDOW_TITLE = (
    f"KNN Explorer \u2014 Model D "
    f"(k={KNN_N_NEIGHBORS}, {len(CORE_FEATURES)} features, {KNN_METRIC})"
)

# ── Theme colours ────────────────────────────────────────────────────────────
BG = "#1e1e1e"
BG_LIGHT = "#2a2a2a"
FG = "#e0e0e0"
FG_DIM = "#aaaaaa"
GREEN = "#88cc88"
RED = "#ff6b6b"
YELLOW = "#ffcc00"
ACCENT = "#3a7bd5"


# ═══════════════════════════════════════════════════════════════════════════════
#  Data & Model
# ═══════════════════════════════════════════════════════════════════════════════

def train_model_d() -> dict:
    """Train the Model D pipeline and return all artefacts for the GUI.

    Returns a dict with keys: df, int_to_type, type_to_int, pipeline,
    X_train, X_test, y_train, y_test, y_pred_test, split_idx,
    X_train_selected, X_test_selected, selected_names, train_meta,
    test_meta, all_ids, test_ids, misclassified_ids.
    """
    df = common.load_data()
    type_to_int, int_to_type = common.get_label_mapping()

    X_train, X_test, y_train, y_test, split_idx = common.get_train_test_split(
        df, feature_cols=CORE_FEATURES,
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(f_classif, k=SELECTKBEST_K)),
        ("knn", KNeighborsClassifier(
            n_neighbors=KNN_N_NEIGHBORS,
            weights=KNN_WEIGHTS,
            metric=KNN_METRIC,
        )),
    ])
    pipeline.fit(X_train, y_train)

    y_pred_test = pipeline.predict(X_test)

    # Pre-compute transformed features for neighbour / counterfactual lookups
    scaler = pipeline.named_steps["scaler"]
    selector = pipeline.named_steps["selector"]
    selected_mask = selector.get_support()
    selected_names = np.array(CORE_FEATURES)[selected_mask].tolist()

    X_train_selected = selector.transform(scaler.transform(X_train))
    X_test_selected = selector.transform(scaler.transform(X_test))

    train_meta = df.iloc[split_idx["train_idx"]].reset_index(drop=True)
    test_meta = df.iloc[split_idx["test_idx"]].reset_index(drop=True)

    all_ids = sorted(df["id"].tolist())
    test_ids = sorted(test_meta["id"].tolist())

    mis_mask = y_pred_test != y_test.values
    misclassified_ids = sorted(test_meta.loc[mis_mask, "id"].tolist())

    return {
        "df": df,
        "int_to_type": int_to_type,
        "type_to_int": type_to_int,
        "pipeline": pipeline,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "split_idx": split_idx,
        "X_train_selected": X_train_selected,
        "X_test_selected": X_test_selected,
        "selected_names": selected_names,
        "train_meta": train_meta,
        "test_meta": test_meta,
        "all_ids": all_ids,
        "test_ids": test_ids,
        "misclassified_ids": misclassified_ids,
    }


def query_pokemon(m: dict, pokemon_id: int) -> Optional[dict]:
    """Look up a Pokemon by Pokedex ID and compute all explanation data.

    Returns a dict with keys: pokemon_id, name, true_type, type2,
    pred_type, correct, in_test, neighbors, vote_counts,
    feat_dist_top10, counterfactual, raw_features.
    Returns None when *pokemon_id* is not found in the dataset.
    """
    df = m["df"]
    row_matches = df[df["id"] == pokemon_id]
    if row_matches.empty:
        return None

    row = row_matches.iloc[0]
    pipeline = m["pipeline"]
    scaler = pipeline.named_steps["scaler"]
    selector = pipeline.named_steps["selector"]
    knn = pipeline.named_steps["knn"]

    # Scale + select the query point (keep as DataFrame to match scaler's fit)
    x_raw = pd.DataFrame([row[CORE_FEATURES].values.astype(float)], columns=CORE_FEATURES)
    x_selected = selector.transform(scaler.transform(x_raw))[0]

    pred_int = int(knn.predict(x_selected.reshape(1, -1))[0])
    pred_label = m["int_to_type"][pred_int]
    true_label = row["type1"]

    in_test = pokemon_id in m["test_ids"]

    # ── Neighbours ───────────────────────────────────────────────────────────
    distances, neigh_idx = knn.kneighbors(
        x_selected.reshape(1, -1), n_neighbors=KNN_N_NEIGHBORS,
    )
    distances = distances[0]
    neigh_idx = neigh_idx[0]

    neighbors: list[dict] = []
    for ni, dist in zip(neigh_idx, distances):
        nb = m["train_meta"].iloc[ni]
        neighbors.append({
            "name": nb["name"],
            "id": int(nb["id"]),
            "type1": nb["type1"],
            "type2": nb["type2"] if pd.notna(nb["type2"]) else None,
            "distance": float(dist),
            "features_selected": m["X_train_selected"][ni],
        })

    # Vote distribution (raw counts)
    vote_counts: dict[str, int] = {}
    for nb in neighbors:
        vote_counts[nb["type1"]] = vote_counts.get(nb["type1"], 0) + 1

    # ── Feature distance to nearest neighbour (top 10, selected space) ───────
    nearest_feat = m["X_train_selected"][neigh_idx[0]]
    feat_dist = np.abs(x_selected - nearest_feat)
    top10 = np.argsort(feat_dist)[::-1][:10]
    sel_names = m["selected_names"]
    feat_dist_top10 = [(sel_names[i], float(feat_dist[i])) for i in top10]

    # ── Counterfactual: nearest enemy (different type than predicted) ────────
    y_train_arr = m["y_train"].values
    enemy_mask = y_train_arr != pred_int
    enemy_pos = np.where(enemy_mask)[0]
    # Use Manhattan (L1) to stay consistent with the KNN metric
    enemy_dists = np.abs(
        m["X_train_selected"][enemy_pos] - x_selected,
    ).sum(axis=1)
    best = int(np.argmin(enemy_dists))
    cf_pos = enemy_pos[best]
    cf_meta = m["train_meta"].iloc[cf_pos]
    cf_feat = m["X_train_selected"][cf_pos]

    cf_diff = np.abs(cf_feat - x_selected)
    top3 = np.argsort(cf_diff)[::-1][:3]
    cf_top3 = [(sel_names[i], float(cf_diff[i])) for i in top3]

    return {
        "pokemon_id": pokemon_id,
        "name": row["name"],
        "true_type": true_label,
        "type2": row["type2"] if pd.notna(row["type2"]) else None,
        "pred_type": pred_label,
        "correct": pred_label == true_label,
        "in_test": in_test,
        "neighbors": neighbors,
        "vote_counts": vote_counts,
        "feat_dist_top10": feat_dist_top10,
        "counterfactual": {
            "name": cf_meta["name"],
            "id": int(cf_meta["id"]),
            "type1": cf_meta["type1"],
            "distance": float(enemy_dists[best]),
            "top3_diffs": cf_top3,
        },
        "raw_features": {f: float(row[f]) for f in CORE_FEATURES},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure builders (dark theme, matplotlib -> Tk canvas)
# ═══════════════════════════════════════════════════════════════════════════════

def _load_sprite(pokemon_id: int, size: tuple[int, int] | None = None) -> Image.Image | None:
    """Load a sprite as RGBA composited onto dark background, optionally resized."""
    path = common.get_sprite_path(pokemon_id)
    if not path.exists():
        return None
    img = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", img.size, (30, 30, 30, 255))
    bg.paste(img, mask=img.split()[3])
    if size is not None:
        bg = bg.resize(size, Image.NEAREST)
    return bg


def build_neighbor_panel(neighbors: list[dict], pred_type: str) -> plt.Figure:
    """Horizontal row of neighbour sprites with type-coloured borders.

    Neighbours whose type matches *pred_type* get a thicker, brighter
    border to indicate they voted for the winning class.
    """
    k = len(neighbors)
    fig, axes = plt.subplots(1, k, figsize=(k * 1.9, 2.6), squeeze=False)
    fig.patch.set_facecolor(BG)
    axes_flat = axes[0]

    for ax, nb in zip(axes_flat, neighbors):
        ax.set_facecolor(BG)
        sprite = _load_sprite(nb["id"])
        if sprite is not None:
            ax.imshow(sprite)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color=FG)

        tc = common.TYPE_COLORS.get(nb["type1"], "#888888")
        voted_pred = nb["type1"] == pred_type
        lw = 3.5 if voted_pred else 1.2
        for spine in ax.spines.values():
            spine.set_edgecolor(tc)
            spine.set_linewidth(lw)

        ax.set_title(
            f"{nb['name'].capitalize()}\n{nb['type1']}  d={nb['distance']:.2f}",
            fontsize=7,
            color=tc,
            fontweight="bold" if voted_pred else "normal",
        )
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("K-Nearest Neighbors", fontsize=10, color=FG, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return fig


def build_confidence_bar(
    vote_counts: dict[str, int],
    pred_type: str,
    k: int,
) -> plt.Figure:
    """Horizontal stacked bar showing the neighbour-vote distribution."""
    fig, ax = plt.subplots(figsize=(4, 0.9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    sorted_types = sorted(
        vote_counts, key=lambda t: (t != pred_type, -vote_counts[t]),
    )
    left = 0.0
    for t in sorted_types:
        cnt = vote_counts[t]
        w = cnt / k
        color = common.TYPE_COLORS.get(t, "#888888")
        ax.barh(0, w, left=left, color=color, edgecolor=BG, height=0.6)
        if w >= 0.10:
            ax.text(
                left + w / 2, 0, f"{cnt}/{k} {t}",
                ha="center", va="center", fontsize=7,
                color="white", fontweight="bold",
            )
        left += w

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    ax.set_title("Vote Distribution", fontsize=8, color=FG, pad=2)
    fig.tight_layout(pad=0.3)
    return fig


def build_feature_distance_chart(
    feat_dist_top10: list[tuple[str, float]],
) -> plt.Figure:
    """Horizontal bar chart of top-10 features by distance to nearest neighbour."""
    names = [n for n, _ in feat_dist_top10][::-1]
    values = [v for _, v in feat_dist_top10][::-1]

    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG_LIGHT)

    max_v = max(values) if values else 1.0
    colors = [
        "#cc4444" if v == max_v
        else "#ee9966" if v >= max_v * 0.65
        else "#8899cc"
        for v in values
    ]
    ax.barh(names, values, color=colors, edgecolor=BG, height=0.65)
    ax.set_title(
        "Feature Distance to Nearest Neighbor",
        fontsize=9, color=FG, pad=4,
    )
    ax.set_xlabel("|query \u2212 neighbor| (scaled)", fontsize=7, color=FG_DIM)
    ax.tick_params(axis="both", labelsize=7, colors=FG_DIM)
    for spine in ax.spines.values():
        spine.set_color("#444444")
    fig.tight_layout(pad=0.5)
    return fig


def build_counterfactual_panel(cf: dict, pred_type: str) -> plt.Figure:
    """Nearest-enemy sprite + top-3 feature differences."""
    fig, (ax_spr, ax_bar) = plt.subplots(
        1, 2, figsize=(5.4, 2.4),
        gridspec_kw={"width_ratios": [1, 2]},
    )
    fig.patch.set_facecolor(BG)

    # ── Sprite ───────────────────────────────────────────────────────────────
    ax_spr.set_facecolor(BG)
    sprite = _load_sprite(cf["id"])
    if sprite is not None:
        ax_spr.imshow(sprite)
    ax_spr.axis("off")
    tc = common.TYPE_COLORS.get(cf["type1"], "#888888")
    ax_spr.set_title(
        f"Nearest Enemy\n{cf['name'].capitalize()} ({cf['type1']})\n"
        f"dist = {cf['distance']:.3f}",
        fontsize=7, color=tc,
    )

    # ── Top-3 feature diffs ──────────────────────────────────────────────────
    ax_bar.set_facecolor(BG_LIGHT)
    bar_names = [n for n, _ in cf["top3_diffs"]][::-1]
    bar_vals = [d for _, d in cf["top3_diffs"]][::-1]
    ax_bar.barh(bar_names, bar_vals, color="#F44336", alpha=0.85, height=0.5)
    ax_bar.set_title("Top-3 Feature Differences", fontsize=8, color=FG, pad=2)
    ax_bar.set_xlabel("|diff| (scaled)", fontsize=7, color=FG_DIM)
    ax_bar.tick_params(axis="both", labelsize=7, colors=FG_DIM)
    for spine in ax_bar.spines.values():
        spine.set_color("#444444")

    fig.suptitle(
        f"Counterfactual \u2014 nearest non-{pred_type} training sample",
        fontsize=8, color=FG_DIM, y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return fig


def build_color_swatches(features: dict[str, float]) -> plt.Figure:
    """Horizontal strip showing dom1 and dom2 colour proportions."""
    fig, ax = plt.subplots(figsize=(2.6, 0.7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    x = 0.0
    for i in (1, 2):
        h = features.get(f"dom{i}_h", 0) / 360.0
        s = features.get(f"dom{i}_s", 0)
        v = features.get(f"dom{i}_v", 0)
        prop = features.get(f"dom{i}_prop", 0)
        color = np.clip(mcolors.hsv_to_rgb([h, s, v]), 0.0, 1.0)
        ax.bar(x + prop / 2, 1, width=prop, color=color,
               edgecolor=BG, linewidth=0.5)
        x += prop

    ax.set_xlim(0, max(x, 0.01))
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Dominant Colors", color=FG_DIM, fontsize=7, pad=2)
    fig.tight_layout(pad=0.2)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  GUI
# ═══════════════════════════════════════════════════════════════════════════════

class KNNExplorer:
    """Interactive KNN explainability GUI aligned with notebook Model D."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1500x920")
        self.root.minsize(1200, 700)
        self.root.configure(bg=BG)

        self.model: Optional[dict] = None
        self.current_id: Optional[int] = None
        self._canvas_refs: list[FigureCanvasTkAgg] = []

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(50, self._load_model)

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        """Create the top-bar controls and the three-column main area."""
        # ── Top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill=tk.X, padx=10, pady=6)

        tk.Label(
            top, text="Pok\u00e9mon ID:", bg=BG, fg=FG, font=("Arial", 11),
        ).pack(side=tk.LEFT)

        self.id_entry = tk.Entry(top, width=7, font=("Arial", 11))
        self.id_entry.pack(side=tk.LEFT, padx=(4, 8))
        self.id_entry.bind("<Return>", lambda _: self._explain_from_entry())

        tk.Button(
            top, text="Explain", command=self._explain_from_entry,
            bg=ACCENT, fg="white", font=("Arial", 10, "bold"),
            relief=tk.FLAT, padx=8,
        ).pack(side=tk.LEFT, padx=(0, 12))

        # Navigation arrows
        tk.Button(
            top, text="\u25c0", command=self._prev,
            bg="#444", fg=FG, font=("Arial", 12), relief=tk.FLAT, width=3,
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            top, text="\u25b6", command=self._next,
            bg="#444", fg=FG, font=("Arial", 12), relief=tk.FLAT, width=3,
        ).pack(side=tk.LEFT, padx=(2, 10))

        tk.Button(
            top, text="Random", command=self._random,
            bg="#555", fg=FG, font=("Arial", 10), relief=tk.FLAT, padx=6,
        ).pack(side=tk.LEFT, padx=2)
        tk.Button(
            top, text="Random Misclassified", command=self._random_misclassified,
            bg="#8B0000", fg="white", font=("Arial", 10), relief=tk.FLAT, padx=6,
        ).pack(side=tk.LEFT, padx=(2, 10))

        # Mode toggle
        self.mode_var = tk.StringVar(value="Test set only")
        mode_combo = ttk.Combobox(
            top, textvariable=self.mode_var,
            values=["Test set only", "All Pok\u00e9mon"],
            state="readonly", width=14, font=("Arial", 10),
        )
        mode_combo.pack(side=tk.LEFT, padx=(0, 12))

        self.status_label = tk.Label(
            top, text="Loading model\u2026", bg=BG, fg=FG_DIM, font=("Arial", 10),
        )
        self.status_label.pack(side=tk.LEFT, padx=8)

        # ── Main area: three columns ────────────────────────────────────────
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 6))

        self.left_frame = tk.Frame(main, bg=BG, width=265)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        self.left_frame.pack_propagate(False)

        self.center_frame = tk.Frame(main, bg=BG)
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        self.right_frame = tk.Frame(main, bg=BG, width=390)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.right_frame.pack_propagate(False)

        # Keyboard shortcuts
        self.root.bind("<Left>", lambda _: self._prev())
        self.root.bind("<Right>", lambda _: self._next())
        self.root.bind("<r>", lambda _: self._random())
        self.root.bind("<m>", lambda _: self._random_misclassified())

    # ── Model loading ────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Train Model D; update status label when done."""
        try:
            self.model = train_model_d()
            n = len(self.model["df"])
            nt = len(self.model["test_ids"])
            nm = len(self.model["misclassified_ids"])
            self.status_label.config(
                text=f"Ready \u2014 {n} Pok\u00e9mon | {nt} test | {nm} misclassified",
                fg=GREEN,
            )
        except Exception as exc:
            self.status_label.config(text=f"Error: {exc}", fg=RED)

    def _on_close(self) -> None:
        plt.close("all")
        self.root.destroy()

    # ── Navigation helpers ───────────────────────────────────────────────────

    def _id_list(self) -> list[int]:
        """Return the sorted ID list for the current mode."""
        if self.model is None:
            return []
        if self.mode_var.get() == "Test set only":
            return self.model["test_ids"]
        return self.model["all_ids"]

    def _navigate(self, direction: int) -> None:
        """Move to the previous (*direction=-1*) or next (*+1*) Pokemon."""
        ids = self._id_list()
        if not ids:
            return
        if self.current_id is None:
            self._explain(ids[0] if direction == 1 else ids[-1])
            return
        # bisect handles the case where current_id isn't in the list (mode switch)
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

    # ── Core explain entry-points ────────────────────────────────────────────

    def _explain_from_entry(self) -> None:
        """Read the ID text field and explain that Pokemon."""
        raw = self.id_entry.get().strip()
        if not raw.isdigit():
            self.status_label.config(text="Enter a numeric Pok\u00e9dex ID.", fg=RED)
            return
        self._explain(int(raw))

    def _explain(self, pokemon_id: int) -> None:
        """Run the full query and render all panels."""
        if self.model is None:
            self.status_label.config(text="Model not loaded yet.", fg=RED)
            return

        result = query_pokemon(self.model, pokemon_id)
        if result is None:
            self.status_label.config(text=f"ID {pokemon_id} not found.", fg=RED)
            return

        self.current_id = pokemon_id
        self.id_entry.delete(0, tk.END)
        self.id_entry.insert(0, str(pokemon_id))
        self._clear_panels()
        self._render(result)

        tag = "test" if result["in_test"] else "train"
        sym = "\u2713" if result["correct"] else "\u2717"
        self.status_label.config(
            text=(
                f"#{pokemon_id} {result['name'].capitalize()} [{tag}]  "
                f"{result['true_type']}\u2192{result['pred_type']} {sym}"
            ),
            fg=GREEN if result["correct"] else RED,
        )

    # ── Panel management ─────────────────────────────────────────────────────

    def _clear_panels(self) -> None:
        for frame in (self.left_frame, self.center_frame, self.right_frame):
            for w in frame.winfo_children():
                w.destroy()
        for c in self._canvas_refs:
            plt.close(c.figure)
        self._canvas_refs.clear()
        plt.close("all")

    def _embed(self, fig: plt.Figure, parent: tk.Frame, **pack_kw) -> None:
        """Draw a matplotlib figure into a Tk frame."""
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(**pack_kw)
        self._canvas_refs.append(canvas)

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render(self, r: dict) -> None:
        """Populate all three columns from a query result dict."""
        self._render_header(r)
        self._render_neighbors(r)
        self._render_counterfactual(r)
        self._render_feature_chart(r)
        self._render_stats(r)

    def _render_header(self, r: dict) -> None:
        """Left column: sprite, name/ID, types, correctness, swatches, votes."""
        f = self.left_frame

        # Sprite
        sprite = _load_sprite(r["pokemon_id"], size=(160, 160))
        if sprite is not None:
            photo = ImageTk.PhotoImage(sprite)
            lbl = tk.Label(f, image=photo, bg=BG)
            lbl.image = photo  # prevent GC
            lbl.pack(pady=(10, 4))

        # Name + ID
        tk.Label(
            f, text=f"{r['name'].capitalize()}  #{r['pokemon_id']}",
            bg=BG, fg=FG, font=("Arial", 13, "bold"),
        ).pack(pady=(2, 4))

        # True type (coloured)
        true_c = common.TYPE_COLORS.get(r["true_type"], FG)
        tk.Label(
            f, text=f"True: {r['true_type']}",
            bg=BG, fg=true_c, font=("Arial", 11),
        ).pack()

        # Predicted type + check / cross
        pred_c = common.TYPE_COLORS.get(r["pred_type"], FG)
        sym = "  \u2713 Correct" if r["correct"] else "  \u2717 Wrong"
        sym_c = GREEN if r["correct"] else RED
        pf = tk.Frame(f, bg=BG)
        pf.pack(pady=2)
        tk.Label(pf, text=f"Pred: {r['pred_type']}", bg=BG, fg=pred_c,
                 font=("Arial", 11)).pack(side=tk.LEFT)
        tk.Label(pf, text=sym, bg=BG, fg=sym_c,
                 font=("Arial", 11, "bold")).pack(side=tk.LEFT)

        # Type2 + set membership
        t2 = r["type2"] if r["type2"] else "\u2014"
        tag = "test set" if r["in_test"] else "train set"
        tk.Label(
            f, text=f"Type 2: {t2}  |  {tag}",
            bg=BG, fg=FG_DIM, font=("Arial", 9),
        ).pack(pady=2)

        if r["type2"] and r["pred_type"] == r["type2"]:
            tk.Label(
                f, text="\u2605 Pred matches type2!",
                bg=BG, fg=YELLOW, font=("Arial", 9, "bold"),
            ).pack()

        # Dominant-colour swatches
        self._embed(build_color_swatches(r["raw_features"]),
                    f, fill=tk.X, padx=4, pady=(4, 2))

        # Vote-distribution bar
        self._embed(
            build_confidence_bar(r["vote_counts"], r["pred_type"], KNN_N_NEIGHBORS),
            f, fill=tk.X, padx=4, pady=2,
        )

    def _render_neighbors(self, r: dict) -> None:
        """Centre column top: neighbour mosaic."""
        self._embed(
            build_neighbor_panel(r["neighbors"], r["pred_type"]),
            self.center_frame, fill=tk.X, pady=(0, 4),
        )

    def _render_counterfactual(self, r: dict) -> None:
        """Centre column bottom: nearest-enemy panel."""
        self._embed(
            build_counterfactual_panel(r["counterfactual"], r["pred_type"]),
            self.center_frame, fill=tk.X, pady=4,
        )

    def _render_feature_chart(self, r: dict) -> None:
        """Right column top: feature-distance horizontal bars."""
        self._embed(
            build_feature_distance_chart(r["feat_dist_top10"]),
            self.right_frame, fill=tk.X, pady=(0, 4),
        )

    def _render_stats(self, r: dict) -> None:
        """Right column bottom: scrollable table of raw CORE_FEATURES values."""
        tk.Label(
            self.right_frame, text="CORE_FEATURES (raw values)",
            bg=BG, fg=FG_DIM, font=("Arial", 9, "bold"),
        ).pack(anchor=tk.W, padx=4)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Dark.Treeview",
            background=BG_LIGHT, foreground=FG, fieldbackground=BG_LIGHT,
            font=("Courier", 8), rowheight=18,
        )
        style.configure(
            "Dark.Treeview.Heading",
            background="#444", foreground=FG, font=("Arial", 8, "bold"),
        )

        tf = tk.Frame(self.right_frame, bg=BG)
        tf.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        tree = ttk.Treeview(
            tf, columns=("feature", "value"), show="headings",
            style="Dark.Treeview", height=14,
        )
        tree.heading("feature", text="Feature")
        tree.heading("value", text="Value")
        tree.column("feature", width=150, anchor=tk.W)
        tree.column("value", width=90, anchor=tk.E)

        sb = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        for feat, val in r["raw_features"].items():
            tree.insert("", tk.END, values=(feat, f"{val:.4f}"))


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Launch the KNN Explorer."""
    root = tk.Tk()
    KNNExplorer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
