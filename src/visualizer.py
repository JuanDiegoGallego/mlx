"""
visualizer.py — Tkinter GUI for inspecting Pokémon color features.

Displays K-means dominant color swatches and a hue histogram for any
Pokémon, alongside the original sprite. Reads from data/processed/features.csv
and data/raw/{id}.png.
"""

import json
import math
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

# Constants
K_CLUSTERS = 5
NUM_HUE_BINS = 12
HUE_BIN_WIDTH = 360 // NUM_HUE_BINS  # 30°
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 620

PROJECT_ROOT = Path(__file__).parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_CSV = PROJECT_ROOT / "data" / "processed" / "features.csv"
LABELS_JSON = PROJECT_ROOT / "data" / "labels.json"


def hsv_to_hex(h: float, s: float, v: float) -> str:
    """Convert HSV (H in [0,360), S and V in [0,1]) to a hex color string.

    Args:
        h: Hue in degrees [0, 360).
        s: Saturation [0, 1].
        v: Value [0, 1].

    Returns:
        Hex color string like '#rrggbb'.
    """
    r, g, b = mcolors.hsv_to_rgb([h / 360.0, s, v])
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def build_swatch_figure(row: pd.Series) -> plt.Figure:
    """Build a matplotlib figure showing 5 dominant color swatches.

    Each swatch's height is proportional to its cluster proportion.

    Args:
        row: A single row from features.csv.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(2.5, 4))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")

    clusters = []
    for i in range(1, K_CLUSTERS + 1):
        clusters.append({
            "h": row[f"dom{i}_h"],
            "s": row[f"dom{i}_s"],
            "v": row[f"dom{i}_v"],
            "prop": row[f"dom{i}_prop"],
        })

    y = 0.0
    for cluster in clusters:
        height = cluster["prop"]
        color = np.clip(mcolors.hsv_to_rgb([cluster["h"] / 360.0, cluster["s"], cluster["v"]]), 0.0, 1.0)
        ax.barh(y + height / 2, 1, height=height, color=color, edgecolor="none")
        label = (f"H:{cluster['h']:.0f}° S:{cluster['s']:.2f} "
                 f"V:{cluster['v']:.2f}\n{cluster['prop']*100:.1f}%")
        ax.text(0.5, y + height / 2, label, ha="center", va="center",
                fontsize=6, color="white", fontweight="bold")
        y += height

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Dominant Colors", color="white", fontsize=9, pad=4)
    fig.tight_layout(pad=0.3)
    return fig


def build_histogram_figure(row: pd.Series) -> plt.Figure:
    """Build a matplotlib figure showing the hue histogram.

    Each bar is colored with the representative hue for that bin.

    Args:
        row: A single row from features.csv.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2a2a2a")

    bin_centers = [i * HUE_BIN_WIDTH + HUE_BIN_WIDTH // 2 for i in range(NUM_HUE_BINS)]
    proportions = [row[f"hue_bin_{i * HUE_BIN_WIDTH}"] for i in range(NUM_HUE_BINS)]
    colors = [np.clip(mcolors.hsv_to_rgb([h / 360.0, 0.85, 0.9]), 0.0, 1.0) for h in bin_centers]

    ax.bar(range(NUM_HUE_BINS), proportions, color=colors, edgecolor="#1e1e1e", linewidth=0.5)
    ax.set_xticks(range(NUM_HUE_BINS))
    ax.set_xticklabels([f"{i * HUE_BIN_WIDTH}°" for i in range(NUM_HUE_BINS)],
                       rotation=45, fontsize=6, color="white")
    ax.set_ylabel("Proportion", color="white", fontsize=8)
    ax.tick_params(axis="y", colors="white", labelsize=7)
    ax.spines[:].set_color("#444444")
    ax.set_title("Hue Histogram (30° bins)", color="white", fontsize=9, pad=4)
    fig.tight_layout(pad=0.5)
    return fig


class PokemonVisualizer:
    """Main application window for Pokémon color feature inspection."""

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the GUI, load data, and build the layout.

        Args:
            root: The root Tk window.
        """
        self.root = root
        self.root.title("Pokémon Color Feature Inspector")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.resizable(False, False)
        self.root.configure(bg="#1e1e1e")

        self.df: Optional[pd.DataFrame] = None
        self.labels: dict = {}
        self._load_data()
        self._build_ui()

    def _load_data(self) -> None:
        """Load features CSV and labels JSON."""
        if FEATURES_CSV.exists():
            self.df = pd.read_csv(FEATURES_CSV, index_col=False)
            self.df["id"] = self.df["id"].astype(int)
        else:
            messagebox.showwarning(
                "Data not found",
                f"features.csv not found at:\n{FEATURES_CSV}\n\n"
                "Run processer.py first."
            )

        if LABELS_JSON.exists():
            with LABELS_JSON.open(encoding="utf-8") as f:
                self.labels = json.load(f)
        # Invert for decoding
        self.label_names = {v: k for k, v in self.labels.items()}

    def _build_ui(self) -> None:
        """Construct all UI widgets."""
        # Top bar
        top_frame = tk.Frame(self.root, bg="#1e1e1e")
        top_frame.pack(fill=tk.X, padx=10, pady=8)

        tk.Label(top_frame, text="Pokémon ID:", bg="#1e1e1e", fg="white",
                 font=("Arial", 11)).pack(side=tk.LEFT)

        self.id_entry = tk.Entry(top_frame, width=8, font=("Arial", 11))
        self.id_entry.pack(side=tk.LEFT, padx=(6, 10))
        self.id_entry.bind("<Return>", lambda _: self._show_pokemon())

        tk.Button(top_frame, text="Show", command=self._show_pokemon,
                  bg="#3a7bd5", fg="white", font=("Arial", 11),
                  relief=tk.FLAT, padx=10).pack(side=tk.LEFT)

        self.status_label = tk.Label(top_frame, text="", bg="#1e1e1e",
                                     fg="#aaaaaa", font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT, padx=16)

        # Main area
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        # Left: color swatches
        self.left_frame = tk.Frame(main_frame, bg="#1e1e1e", width=260)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        self.left_frame.pack_propagate(False)

        self.swatch_canvas_widget: Optional[FigureCanvasTkAgg] = None

        # Center: hue histogram
        self.center_frame = tk.Frame(main_frame, bg="#1e1e1e", width=380)
        self.center_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        self.center_frame.pack_propagate(False)

        self.hist_canvas_widget: Optional[FigureCanvasTkAgg] = None

        # Right: sprite + info
        self.right_frame = tk.Frame(main_frame, bg="#1e1e1e")
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.sprite_label = tk.Label(self.right_frame, bg="#1e1e1e")
        self.sprite_label.pack(pady=(20, 6))

        self.info_label = tk.Label(self.right_frame, text="", bg="#1e1e1e",
                                   fg="white", font=("Arial", 11),
                                   justify=tk.CENTER)
        self.info_label.pack()

        # Initial placeholder
        self._show_placeholder()

    def _show_placeholder(self) -> None:
        """Show startup instructions."""
        msg = tk.Label(self.left_frame,
                       text="Enter a Pokémon ID\nand click Show",
                       bg="#1e1e1e", fg="#888888",
                       font=("Arial", 10), justify=tk.CENTER)
        msg.pack(expand=True)

    def _clear_panels(self) -> None:
        """Remove all widgets from the three panels."""
        for widget in self.left_frame.winfo_children():
            widget.destroy()
        for widget in self.center_frame.winfo_children():
            widget.destroy()
        for widget in self.right_frame.winfo_children():
            widget.destroy()
        plt.close("all")

    def _show_pokemon(self) -> None:
        """Fetch and display data for the entered Pokémon ID."""
        raw = self.id_entry.get().strip()
        if not raw.isdigit():
            self.status_label.config(text="Please enter a valid numeric ID.", fg="#ff6b6b")
            return

        pokemon_id = int(raw)

        if self.df is None:
            self.status_label.config(text="Data not loaded.", fg="#ff6b6b")
            return

        matches = self.df[self.df["id"] == pokemon_id]
        if matches.empty:
            self.status_label.config(
                text=f"ID {pokemon_id} not found in features.csv.", fg="#ff6b6b"
            )
            return

        row = matches.iloc[0]
        self.status_label.config(text="", fg="#aaaaaa")
        self._clear_panels()
        self._render_swatches(row)
        self._render_histogram(row)
        self._render_sprite_info(row, pokemon_id)

    def _render_swatches(self, row: pd.Series) -> None:
        """Render dominant color swatches in the left panel."""
        fig = build_swatch_figure(row)
        canvas = FigureCanvasTkAgg(fig, master=self.left_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.swatch_canvas_widget = canvas

    def _render_histogram(self, row: pd.Series) -> None:
        """Render hue histogram in the center panel."""
        fig = build_histogram_figure(row)
        canvas = FigureCanvasTkAgg(fig, master=self.center_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.hist_canvas_widget = canvas

    def _render_sprite_info(self, row: pd.Series, pokemon_id: int) -> None:
        """Render sprite image and Pokémon info in the right panel."""
        sprite_path = RAW_DIR / f"{pokemon_id}.png"

        if sprite_path.exists():
            img = Image.open(sprite_path).convert("RGBA")
            # Scale up sprite (96×96 → 192×192) for visibility
            img = img.resize((192, 192), Image.NEAREST)
            photo = ImageTk.PhotoImage(img)
            sprite_lbl = tk.Label(self.right_frame, image=photo, bg="#1e1e1e")
            sprite_lbl.image = photo  # keep reference
            sprite_lbl.pack(pady=(20, 6))
        else:
            tk.Label(self.right_frame, text="[sprite not found]",
                     bg="#1e1e1e", fg="#888888").pack(pady=(20, 6))

        type2_str = row["type2"] if pd.notna(row.get("type2")) else "—"
        info = (
            f"{row['name'].capitalize()}  (#{pokemon_id})\n"
            f"Type 1: {row['type1'].capitalize()}\n"
            f"Type 2: {str(type2_str).capitalize()}"
        )
        tk.Label(self.right_frame, text=info, bg="#1e1e1e", fg="white",
                 font=("Arial", 11), justify=tk.CENTER).pack()


def main() -> None:
    """Launch the Pokémon color feature visualizer."""
    root = tk.Tk()
    app = PokemonVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
