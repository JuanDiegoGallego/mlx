"""
knn_explorer.py — Tkinter GUI for interactive KNN explainability.

Displays a Pokémon sprite, its k nearest neighbours with type-colored borders,
a neighbor type distribution bar chart, and a per-feature distance breakdown.

Trains a lightweight KNN on the pruned feature set (features with positive
permutation importance from the notebook analysis) at startup.
"""

import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import common

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 700

# Best hyperparams from the notebook grid search (KNN + SMOTE pipeline).
# These are hardcoded so the explorer works without re-running the notebook.
KNN_N_NEIGHBORS = 9
KNN_WEIGHTS = "distance"
KNN_METRIC = "manhattan"

# Features with positive permutation importance (from notebook section 3.1).
# If the notebook hasn't been run yet, fall back to all features.
PRUNED_FEATURES: Optional[list[str]] = None  # set at runtime


# ── Model setup ──────────────────────────────────────────────────────────────

def _train_knn():
    """Load data, train a KNN on the pruned feature set, return all artefacts."""
    df = common.load_data()
    type_to_int, int_to_type = common.get_label_mapping()
    X_train, X_test, y_train, y_test, split_idx = common.get_train_test_split(df)

    feature_cols = common.FEATURE_COLS_ALL

    scaler_full = StandardScaler()
    scaler_full.fit(X_train)

    # Use all features for the explanation KNN (no SMOTE, real neighbours)
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), index=X_train.index, columns=feature_cols
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), index=X_test.index, columns=feature_cols
    )

    knn = KNeighborsClassifier(
        n_neighbors=KNN_N_NEIGHBORS, weights=KNN_WEIGHTS, metric=KNN_METRIC
    )
    knn.fit(X_train_sc, y_train)

    y_pred = knn.predict(X_test_sc)

    return {
        "df": df,
        "int_to_type": int_to_type,
        "type_to_int": type_to_int,
        "feature_cols": feature_cols,
        "X_train_sc": X_train_sc,
        "X_test_sc": X_test_sc,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "split_idx": split_idx,
        "knn": knn,
    }


# ── Figure builders ──────────────────────────────────────────────────────────

def build_neighbor_mosaic(df, neighbor_df_indices, query_name, pokemon_id, k):
    """Sprite mosaic of k nearest neighbours with type-colored borders."""
    n_cols = min(k, 5)
    n_rows = (k + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.6, n_rows * 2.0),
        squeeze=False,
    )
    fig.patch.set_facecolor("#1e1e1e")
    axes_flat = axes.reshape(-1)

    for ax_i, ni in enumerate(neighbor_df_indices):
        nb = df.loc[ni]
        sprite_path = common.get_sprite_path(int(nb["id"]))
        ax = axes_flat[ax_i]
        ax.set_facecolor("#1e1e1e")
        if sprite_path.exists():
            img = Image.open(sprite_path).convert("RGBA")
            bg = Image.new("RGBA", img.size, (30, 30, 30, 255))
            bg.paste(img, mask=img.split()[3])
            ax.imshow(bg)
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="white")
        type_color = common.TYPE_COLORS.get(nb["type1"], "#888888")
        for spine in ax.spines.values():
            spine.set_edgecolor(type_color)
            spine.set_linewidth(2.5)
        ax.set_title(
            f"{nb['name'].capitalize()}\n{nb['type1']}",
            fontsize=7, color=type_color,
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for ax_i in range(len(neighbor_df_indices), len(axes_flat)):
        axes_flat[ax_i].axis("off")
        axes_flat[ax_i].set_facecolor("#1e1e1e")

    fig.suptitle(
        f"Neighbours of {query_name} (#{pokemon_id})",
        fontsize=9, color="white",
    )
    fig.tight_layout(pad=0.4, rect=[0, 0, 1, 0.93])
    return fig


def build_type_dist_chart(df, neighbor_df_indices):
    """Bar chart of neighbour type distribution."""
    neighbor_types = [df.loc[ni]["type1"] for ni in neighbor_df_indices]
    type_counts = pd.Series(neighbor_types).value_counts()

    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2a2a2a")
    ax.bar(
        type_counts.index, type_counts.values,
        color=[common.TYPE_COLORS.get(t, "#888") for t in type_counts.index],
        edgecolor="#1e1e1e", linewidth=0.5,
    )
    ax.set_title("Neighbor Types", fontsize=9, color="white", pad=4)
    ax.set_ylabel("Count", fontsize=8, color="white")
    ax.tick_params(axis="x", rotation=30, labelsize=7, colors="white")
    ax.tick_params(axis="y", labelsize=7, colors="white")
    ax.spines[:].set_color("#444444")
    fig.tight_layout(pad=0.5)
    return fig


def build_distance_chart(X_train_sc, neighbor_df_indices, x_query):
    """Horizontal bar chart of per-feature absolute distance to neighbours."""
    neighbor_features = X_train_sc.loc[neighbor_df_indices]
    abs_diffs = (neighbor_features - x_query.values).abs()
    mean_diff = abs_diffs.mean(axis=0).nlargest(12).sort_values()

    bar_colors = [
        "#cc4444" if v == mean_diff.max()
        else "#ee9966" if v >= mean_diff.quantile(0.75)
        else "#8899cc"
        for v in mean_diff.values
    ]

    fig, ax = plt.subplots(figsize=(4, 3.5))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2a2a2a")
    mean_diff.plot.barh(ax=ax, color=bar_colors)
    ax.set_title("Feature Distance (top 12)", fontsize=9, color="white", pad=4)
    ax.set_xlabel("Mean |diff| (z-score)", fontsize=7, color="white")
    ax.tick_params(axis="both", labelsize=7, colors="white")
    ax.spines[:].set_color("#444444")
    fig.tight_layout(pad=0.5)
    return fig


def build_color_swatches(row: pd.Series) -> plt.Figure:
    """Horizontal strip of 5 dominant colors, widths proportional to cluster share."""
    fig, ax = plt.subplots(figsize=(2.0, 0.9))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    x = 0.0
    for i in range(1, 6):
        prop = float(row[f"dom{i}_prop"])
        color = np.clip(
            mcolors.hsv_to_rgb(
                [float(row[f"dom{i}_h"]) / 360.0, float(row[f"dom{i}_s"]), float(row[f"dom{i}_v"])]
            ),
            0.0, 1.0,
        )
        ax.bar(x + prop / 2, 1, width=prop, color=color, edgecolor="#1e1e1e", linewidth=0.5)
        x += prop

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Dominant Colors", color="white", fontsize=7, pad=2)
    fig.tight_layout(pad=0.2)
    return fig


# ── Main application ─────────────────────────────────────────────────────────

class KNNExplorer:
    """Interactive KNN explainability GUI."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("KNN Pokémon Type Explorer — XAI")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e1e")

        self.status_label: Optional[tk.Label] = None
        self._build_ui()

        # Train model in background after UI is drawn
        self.model = None
        self.root.after(100, self._load_model)

    def _build_ui(self) -> None:
        # ── Top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg="#1e1e1e")
        top.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(top, text="Pokémon ID:", bg="#1e1e1e", fg="white",
                 font=("Arial", 11)).pack(side=tk.LEFT)
        self.id_entry = tk.Entry(top, width=8, font=("Arial", 11))
        self.id_entry.pack(side=tk.LEFT, padx=(6, 10))
        self.id_entry.bind("<Return>", lambda _: self._explain())

        tk.Button(top, text="Explain", command=self._explain,
                  bg="#3a7bd5", fg="white", font=("Arial", 11),
                  relief=tk.FLAT, padx=10).pack(side=tk.LEFT)

        self.status_label = tk.Label(top, text="Loading model...",
                                     bg="#1e1e1e", fg="#aaaaaa",
                                     font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, padx=16)

        # ── Main area ────────────────────────────────────────────────────────
        main = tk.Frame(self.root, bg="#1e1e1e")
        main.pack(fill=tk.BOTH, expand=True, padx=10)

        # Left: sprite + info
        self.left_frame = tk.Frame(main, bg="#1e1e1e", width=220)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        self.left_frame.pack_propagate(False)

        # Center: neighbour mosaic
        self.center_frame = tk.Frame(main, bg="#1e1e1e", width=480)
        self.center_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        self.center_frame.pack_propagate(False)

        # Right: charts (stacked)
        self.right_frame = tk.Frame(main, bg="#1e1e1e")
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _load_model(self) -> None:
        """Train the KNN model."""
        try:
            self.model = _train_knn()
            self.status_label.config(
                text=f"Ready — {len(self.model['df'])} Pokémon loaded",
                fg="#88cc88",
            )
        except Exception as e:
            self.status_label.config(text=f"Error: {e}", fg="#ff6b6b")

    def _clear_panels(self) -> None:
        for frame in (self.left_frame, self.center_frame, self.right_frame):
            for w in frame.winfo_children():
                w.destroy()
        plt.close("all")

    def _explain(self) -> None:
        raw = self.id_entry.get().strip()
        if not raw.isdigit():
            self.status_label.config(text="Enter a numeric ID.", fg="#ff6b6b")
            return
        if self.model is None:
            self.status_label.config(text="Model not loaded yet.", fg="#ff6b6b")
            return

        pokemon_id = int(raw)
        m = self.model
        df = m["df"]

        row_mask = df["id"] == pokemon_id
        if not row_mask.any():
            self.status_label.config(
                text=f"ID {pokemon_id} not found.", fg="#ff6b6b"
            )
            return

        row = df[row_mask].iloc[0]
        df_pos = df[row_mask].index[0]
        test_indices = m["split_idx"]["test_idx"]
        match = np.where(test_indices == np.where(df.index == df_pos)[0][0])[0]

        # Allow both train and test set Pokémon
        if len(match) > 0:
            i = match[0]
            x_query = m["X_test_sc"].iloc[i]
            pred_int = int(m["y_pred"][i])
            in_test = True
        else:
            # Training set — predict directly
            train_indices = m["split_idx"]["train_idx"]
            match_tr = np.where(
                train_indices == np.where(df.index == df_pos)[0][0]
            )[0]
            if len(match_tr) == 0:
                self.status_label.config(
                    text=f"ID {pokemon_id} not in splits.", fg="#ff6b6b"
                )
                return
            i_tr = match_tr[0]
            x_query = m["X_train_sc"].iloc[i_tr]
            pred_int = int(m["knn"].predict(x_query.values.reshape(1, -1))[0])
            in_test = False

        pred_label = m["int_to_type"][pred_int]
        true_label = row["type1"]
        type2 = row["type2"] if pd.notna(row["type2"]) else "—"

        self._clear_panels()
        self.status_label.config(text="", fg="#aaaaaa")

        # ── Left: sprite + info ─────────────────────────────────────────────
        sprite_path = common.get_sprite_path(pokemon_id)
        if sprite_path.exists():
            img = Image.open(sprite_path).convert("RGBA")
            img = img.resize((160, 160), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(self.left_frame, image=photo, bg="#1e1e1e")
            lbl.image = photo
            lbl.pack(pady=(20, 6))

        correct = pred_label == true_label
        color = "#88cc88" if correct else "#ff6b6b"
        info = (
            f"{row['name'].capitalize()}  (#{pokemon_id})\n\n"
            f"True:      {true_label}\n"
            f"Predicted: {pred_label}  {'✓' if correct else '✗'}\n"
            f"Type 2:    {type2}\n"
            f"Set:       {'test' if in_test else 'train'}"
        )
        tk.Label(self.left_frame, text=info, bg="#1e1e1e", fg=color,
                 font=("Courier", 10), justify=tk.LEFT).pack(pady=6)

        if type2 != "—" and pred_label == type2:
            tk.Label(
                self.left_frame,
                text="★ Pred matches type2!",
                bg="#1e1e1e", fg="#ffcc00",
                font=("Arial", 9, "bold"),
            ).pack()

        # Color swatches
        fig_swatches = build_color_swatches(row)
        canvas_sw = FigureCanvasTkAgg(fig_swatches, master=self.left_frame)
        canvas_sw.draw()
        canvas_sw.get_tk_widget().pack(fill=tk.X, padx=4, pady=(6, 2))

        # ── Neighbours ───────────────────────────────────────────────────────
        knn = m["knn"]
        distances, neigh_idx = knn.kneighbors(
            x_query.values.reshape(1, -1), n_neighbors=knn.n_neighbors
        )
        neighbor_df_indices = m["X_train_sc"].index[neigh_idx[0]]

        # Center: mosaic
        fig_mosaic = build_neighbor_mosaic(
            df, neighbor_df_indices,
            row["name"].capitalize(), pokemon_id, knn.n_neighbors,
        )
        canvas = FigureCanvasTkAgg(fig_mosaic, master=self.center_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right top: type dist
        fig_dist = build_type_dist_chart(df, neighbor_df_indices)
        canvas2 = FigureCanvasTkAgg(fig_dist, master=self.right_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.X, pady=(0, 4))

        # Right bottom: feature distance
        fig_feat = build_distance_chart(
            m["X_train_sc"], neighbor_df_indices, x_query
        )
        canvas3 = FigureCanvasTkAgg(fig_feat, master=self.right_frame)
        canvas3.draw()
        canvas3.get_tk_widget().pack(fill=tk.X)


def main() -> None:
    """Launch the KNN Pokémon Type Explorer."""
    root = tk.Tk()
    app = KNNExplorer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
