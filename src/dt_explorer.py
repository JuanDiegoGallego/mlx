"""
dt_explorer.py — Tkinter GUI for interactive Decision Tree explainability.

Displays a Pokémon sprite, SHAP waterfall explanation, feature importance,
and prediction confidence.
"""

import sys
import tkinter as tk
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import common

# ── Constants ────────────────────────────────────────────────────────────────
WINDOW_WIDTH = 1500
WINDOW_HEIGHT = 750

# DT hyperparams from notebook
DT_MAX_DEPTH = 15
DT_MIN_SAMPLES_LEAF = 1
DT_MIN_SAMPLES_SPLIT = 2
DT_CRITERION = "gini"
DT_MAX_FEATURES = None


# ── Model setup ──────────────────────────────────────────────────────────────

def _train_dt():
    """Load data, train DT with SMOTE, return artefacts."""
    df = common.load_data()
    type_to_int, int_to_type = common.get_label_mapping()
    feature_cols = common.FEATURE_COLS_ALL

    X = df[feature_cols].copy()
    y = df["type1"].map(type_to_int).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE on train only
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    # Scale - preserve indices for both train and test
    scaler = StandardScaler()
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_train_sm_sc = pd.DataFrame(
        scaler.transform(X_train_sm), columns=feature_cols, index=X_train_sm.index
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    # Train DT on SMOTE data
    dt = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH,
        min_samples_leaf=DT_MIN_SAMPLES_LEAF,
        min_samples_split=DT_MIN_SAMPLES_SPLIT,
        criterion=DT_CRITERION,
        max_features=DT_MAX_FEATURES,
        class_weight="balanced",
        random_state=42,
    )
    dt.fit(X_train_sm_sc, y_train_sm)

    y_pred = dt.predict(X_test_sc)
    y_prob = dt.predict_proba(X_test_sc)

    # SHAP for both test and train
    shap_explainer = shap.TreeExplainer(dt)
    shap_values_test = shap_explainer.shap_values(X_test_sc)
    shap_values_train = shap_explainer.shap_values(X_train_sc)

    return {
        "df": df,
        "int_to_type": int_to_type,
        "type_to_int": type_to_int,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_train_sc": X_train_sc,
        "X_test": X_test,
        "X_test_sc": X_test_sc,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "dt": dt,
        "scaler": scaler,
        "shap_values_test": shap_values_test,
        "shap_values_train": shap_values_train,
        "shap_explainer": shap_explainer,
    }


# ── Figure builders ──────────────────────────────────────────────────────────

def build_shap_waterfall(dt, shap_values, shap_explainer, X_test_sc, test_idx, feature_cols):
    """SHAP waterfall for predicted class."""
    pred_int = int(dt.predict(X_test_sc.iloc[[test_idx]])[0])
    pred_idx = list(dt.classes_).index(pred_int)

    sv = shap_values[pred_idx][test_idx] if isinstance(shap_values, list) else shap_values[test_idx]
    base = shap_explainer.expected_value[pred_idx] if hasattr(
        shap_explainer.expected_value, "__len__"
    ) else shap_explainer.expected_value

    exp = shap.Explanation(
        values=sv, base_values=base,
        data=X_test_sc.iloc[test_idx].values,
        feature_names=feature_cols,
    )

    fig = plt.figure(figsize=(5, 2.2), dpi=80)
    shap.plots.waterfall(exp, show=False)
    fig.patch.set_facecolor("#1e1e1e")

    # Style ALL axes and text for dark theme
    for ax in fig.get_axes():
        ax.set_facecolor("#1e1e1e")
        for spine in ax.spines.values():
            spine.set_color("white")
        ax.tick_params(colors="white", labelcolor="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        if ax.get_title():
            ax.title.set_color("white")

    # Catch every text object SHAP created (annotations, y-labels, title)
    for text_obj in fig.findobj(plt.Text):
        text_obj.set_color("white")

    fig.subplots_adjust(left=0.35)
    fig.tight_layout()
    return fig

def build_top_features_chart(dt, shap_values, X_test_sc, shap_explainer, feature_cols, test_idx):
    """Bar chart of top SHAP features."""
    pred_int = int(dt.predict(X_test_sc.iloc[[test_idx]])[0])
    pred_idx = list(dt.classes_).index(pred_int)

    sv = shap_values[pred_idx][test_idx] if isinstance(shap_values, list) else shap_values[test_idx]
    top_features = pd.Series(np.abs(sv), index=feature_cols).nlargest(10)

    # Map each feature to its color based on SHAP sign
    color_map = {ft: ("#4a90e2" if sv[feature_cols.index(ft)] < 0 else "#ff6b6b")
                 for ft in top_features.index}

    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2a2a2a")

    top_features_sorted = top_features.sort_values()
    colors = [color_map[ft] for ft in top_features_sorted.index]
    top_features_sorted.plot.barh(ax=ax, color=colors, edgecolor="#1e1e1e")

    ax.set_title("Top SHAP Features", fontsize=9, color="white")
    ax.set_xlabel("|SHAP value|", fontsize=8, color="white")
    ax.tick_params(axis="both", labelsize=7, colors="white")
    ax.spines[:].set_color("#444444")
    fig.tight_layout()
    return fig


def build_prediction_confidence(y_prob_row, int_to_type, dt):
    """Bar chart of top-5 probabilities."""
    top_5_idx = np.argsort(y_prob_row)[::-1][:5]
    top_5_probs = [y_prob_row[i] for i in top_5_idx]
    top_5_types = [int_to_type[dt.classes_[i]] for i in top_5_idx]
    colors = [common.TYPE_COLORS.get(t, "#888") for t in top_5_types]

    fig, ax = plt.subplots(figsize=(5, 3))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2a2a2a")

    bars = ax.barh(top_5_types[::-1], [p * 100 for p in top_5_probs[::-1]],
                   color=colors[::-1], edgecolor="#1e1e1e", linewidth=0.5)
    ax.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8, color="white")

    ax.set_xlim(0, 100)
    ax.set_title("Prediction Confidence", fontsize=9, color="white")
    ax.set_xlabel("Probability (%)", fontsize=8, color="white")
    ax.tick_params(axis="both", labelsize=7, colors="white")
    ax.spines[:].set_color("#444444")
    fig.tight_layout()
    return fig


def build_color_swatches(row: pd.Series) -> plt.Figure:
    """Dominant colors strip."""
    fig, ax = plt.subplots(figsize=(2.5, 0.8))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    x = 0.0
    for i in range(1, 6):
        prop = float(row[f"dom{i}_prop"])
        color = np.clip(
            mcolors.hsv_to_rgb(
                [float(row[f"dom{i}_h"]) / 360.0, float(row[f"dom{i}_s"]),
                 float(row[f"dom{i}_v"])]
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

class DTExplorer:
    """Interactive Decision Tree explainability GUI."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Decision Tree Pokémon Type Explorer — XAI")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e1e")

        self.status_label: Optional[tk.Label] = None
        self._build_ui()

        self.model = None
        self.root.after(100, self._load_model)

    def _build_ui(self) -> None:
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

        main = tk.Frame(self.root, bg="#1e1e1e")
        main.pack(fill=tk.BOTH, expand=True, padx=10)

        self.left_frame = tk.Frame(main, bg="#1e1e1e", width=200)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        self.left_frame.pack_propagate(False)

        self.center_frame = tk.Frame(main, bg="#1e1e1e", width=500)
        self.center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))
        self.center_frame.pack_propagate(False)

        self.right_frame = tk.Frame(main, bg="#1e1e1e")
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def _load_model(self) -> None:
        try:
            self.model = _train_dt()
            self.status_label.config(
                text=f"Ready — {len(self.model['df'])} Pokémon",
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
            self.status_label.config(text="Enter numeric ID.", fg="#ff6b6b")
            return
        if self.model is None:
            self.status_label.config(text="Model not loaded.", fg="#ff6b6b")
            return

        pokemon_id = int(raw)
        m = self.model
        df = m["df"]

        row_mask = df["id"] == pokemon_id
        if not row_mask.any():
            self.status_label.config(text=f"ID {pokemon_id} not found.", fg="#ff6b6b")
            return

        row = df[row_mask].iloc[0]
        df_pos = df[row_mask].index[0]
        
        # Find in test or train set
        test_indices = m["X_test"].index
        train_indices = m["X_train"].index
        in_test = False
        test_idx = None
        
        if df_pos in test_indices:
            test_idx = test_indices.get_loc(df_pos)
            pred_int = int(m["y_pred"][test_idx])
            y_prob_row = m["y_prob"][test_idx]
            in_test = True
        elif df_pos in train_indices:
            # Predict on train set
            train_idx = train_indices.get_loc(df_pos)
            x_scaled = m["X_train_sc"].iloc[[train_idx]]
            pred_int = int(m["dt"].predict(x_scaled)[0])
            y_prob_row = m["dt"].predict_proba(x_scaled)[0]
            in_test = False
        else:
            self.status_label.config(text=f"ID {pokemon_id} not in any split.", fg="#ff6b6b")
            return

        pred_label = m["int_to_type"][pred_int]
        true_label = row["type1"]
        type2 = row["type2"] if pd.notna(row["type2"]) else "—"

        self._clear_panels()
        self.status_label.config(text="", fg="#aaaaaa")

        # ── Left: sprite + info ──────────────────────────────────────────────
        sprite_path = common.get_sprite_path(pokemon_id)
        if sprite_path.exists():
            img = Image.open(sprite_path).convert("RGBA")
            img = img.resize((150, 150), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(self.left_frame, image=photo, bg="#1e1e1e")
            lbl.image = photo
            lbl.pack(pady=(20, 6))

        correct = pred_label == true_label
        color = "#88cc88" if correct else "#ff6b6b"
        max_prob = y_prob_row.max()

        info = (
            f"{row['name'].capitalize()}\n(#{pokemon_id})\n\n"
            f"True:      {true_label}\n"
            f"Predicted: {pred_label}\n"
            f"{'✓' if correct else '✗'}\n\n"
            f"Confidence:\n{max_prob*100:.1f}%\n\n"
            f"Type 2:    {type2}"
        )
        tk.Label(self.left_frame, text=info, bg="#1e1e1e", fg=color,
                 font=("Courier", 9), justify=tk.LEFT).pack(pady=6)

        if type2 != "—" and pred_label == type2:
            tk.Label(
                self.left_frame,
                text="★ Matches type 2",
                bg="#1e1e1e", fg="#ffcc00",
                font=("Arial", 8, "bold"),
            ).pack()

        # Data source info
        source_text = "From: TEST" if in_test else "From: TRAIN"
        tk.Label(self.left_frame, text=source_text, bg="#1e1e1e", fg="#aaaaaa",
                 font=("Arial", 7)).pack(pady=(4, 0))

        fig_swatches = build_color_swatches(row)
        canvas_sw = FigureCanvasTkAgg(fig_swatches, master=self.left_frame)
        canvas_sw.draw()
        canvas_sw.get_tk_widget().pack(fill=tk.X, padx=2, pady=(8, 2))

        # ── Center: SHAP waterfall ──────────────────────────────────────────
        if in_test:
            X_for_shap = m["X_test_sc"]
            shap_vals = m["shap_values_test"]
            shap_idx = test_idx
        else:
            train_indices = m["X_train"].index
            train_idx = train_indices.get_loc(df_pos)
            X_for_shap = m["X_train_sc"]
            shap_vals = m["shap_values_train"]
            shap_idx = train_idx

        fig_waterfall = build_shap_waterfall(
            m["dt"], shap_vals, m["shap_explainer"],
            X_for_shap, shap_idx, m["feature_cols"]
        )
        canvas_wf = FigureCanvasTkAgg(fig_waterfall, master=self.center_frame)
        canvas_wf.draw()
        canvas_wf.get_tk_widget().pack(fill=tk.X, anchor="n")

        # ── Right: stacked charts ────────────────────────────────────────────
        fig_feat = build_top_features_chart(
            m["dt"], shap_vals, X_for_shap,
            m["shap_explainer"], m["feature_cols"], shap_idx
        )
        canvas_feat = FigureCanvasTkAgg(fig_feat, master=self.right_frame)
        canvas_feat.draw()
        canvas_feat.get_tk_widget().pack(fill=tk.X, pady=(0, 4))

        fig_conf = build_prediction_confidence(y_prob_row, m["int_to_type"], m["dt"])
        canvas_conf = FigureCanvasTkAgg(fig_conf, master=self.right_frame)
        canvas_conf.draw()
        canvas_conf.get_tk_widget().pack(fill=tk.X)


def main() -> None:
    """Launch DT Explorer."""
    root = tk.Tk()
    app = DTExplorer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
