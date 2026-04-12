# Pokémon Type Prediction from Sprite Colors — XAI Project

**Team:** Grifo Amarillo
**Course:** Explainable Machine Learning (Master's degree)

## Overview

This project predicts a Pokémon's **primary type** (18 classes) using **color features** extracted from its front sprite. The primary objective is not maximum accuracy but **explainability**: we apply XAI techniques (SHAP, LIME, feature importance, decision paths) to KNN, Decision Tree, and Neural Network models to understand *why* they make their predictions.

A particularly interesting analysis involves misclassifications where the predicted type matches the Pokémon's *secondary* type — a natural source of XAI insight.

## Project structure

```
pokemon-xai/
├── README.md
├── requirements.txt
├── data/
│   ├── labels.json          # type -> integer mapping
│   ├── raw/                 # sprites + metadata (created by picker.py)
│   └── processed/           # feature CSV (created by processer.py)
├── src/
│   ├── picker.py            # data collection from PokéAPI
│   ├── processer.py         # color feature extraction
│   └── visualizer.py        # tkinter GUI for inspection
└── notebooks/               # XAI analysis notebooks
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the scripts in order from the project root:

### 1. Download data

```bash
python src/picker.py
```

Downloads metadata and front sprites for Pokémon #1–1025 from PokéAPI into `data/raw/`. The script is resumable — already-downloaded sprites are skipped.

### 2. Extract features

```bash
python src/processer.py
```

Reads `data/raw/pokemon.json` and each sprite PNG, extracts 42 color features per Pokémon, and saves `data/processed/features.csv`.

### 3. Inspect features visually

```bash
python src/visualizer.py
```

Opens a GUI where you can enter any Pokémon ID and see:
- K-means dominant color swatches (proportional height)
- Hue histogram (12 bins × 30°)
- Original sprite with name and types

## Feature extraction approach

Each sprite undergoes a 5-step pipeline:

| Step | Description | Features |
|------|-------------|----------|
| 0 | Load PNG, mask transparent background | — |
| 1 | Convert non-transparent pixels to HSV | — |
| 2 | Encode hue as (sin, cos) to handle circularity | — |
| 3 | K-means (k=5) dominant colors | 20 (H, S, V, proportion × 5) |
| 4 | HSV distribution statistics | 10 (means, stds, proportions) |
| 5 | Hue histogram (12 × 30° bins) | 12 |
| **Total** | | **42 features** |

## Notebooks

- `notebooks/0_EDA.ipynb`: Exploratory data analysis of the extracted features.
- `notebooks/1_decision_tree.ipynb`: Decision Tree model training and XAI analysis.
- `notebooks/2_knn.ipynb`: KNN model training and XAI analysis.
- `notebooks/3_neural_network.ipynb`: Neural Network model training and XAI analysis.
