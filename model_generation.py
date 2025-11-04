# %%
# Template guidelines
# - This file works as both a script and an interactive notebook-like workflow.
# - Cells are marked with "#%%". Each cell starts with a one-line description.
# - Parameters are defined as UPPERCASE constants. When run as a script, optional CLI flags can overwrite them.
# - Argument parsing runs only when executed as a script, not in IPython or Jupyter. Overwritten values are written back into globals().
# - Autoreload is enabled automatically in IPython so edits to imported modules take effect without restarting.
# - The example uses scikit-learn to train a simple classifier on the Iris dataset or a user CSV.

# %%
# Imports and autoreload
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython import get_ipython

ipython = get_ipython()
if ipython is not None:
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

# %%
# Global defaults (can be overridden by CLI when run as a script)
DATA_PATH: str | None = None  # Path to a CSV file. If None, use sklearn's Iris dataset.
TARGET_COLUMN: str | None = None  # Required if DATA_PATH is set.
TEST_SIZE: float = 0.2  # Fraction for test split.
RANDOM_STATE: int = 42  # Random seed.
N_ESTIMATORS: int = 200  # Used if ESTIMATOR_GRID is empty.
MAX_DEPTH: int | None = None  # Tree max depth. None means unlimited.
ESTIMATOR_GRID: str = (
    "100,200,300"  # Comma-separated values for n_estimators. Empty string to disable grid.
)
MODEL_OUT: str = "model.joblib"  # Output path for the trained pipeline.
VERBOSE: int = 1  # 0 for quiet prints, 1 for brief status.

# %%
# CLI overrides (only when run as a script, not in IPython/Jupyter)
if __name__ == "__main__" and ipython is None:
    parser = argparse.ArgumentParser(
        description="Cell-friendly sklearn template with optional CLI overrides."
    )
    parser.add_argument(
        "--data-path",
        dest="DATA_PATH",
        type=str,
        default=None,
        help="CSV file to load. If omitted, use the Iris dataset.",
    )
    parser.add_argument(
        "--target",
        dest="TARGET_COLUMN",
        type=str,
        default=None,
        help="Name of the target column in the CSV. Required when --data-path is given.",
    )
    parser.add_argument(
        "--test-size",
        dest="TEST_SIZE",
        type=float,
        default=None,
        help="Test split fraction. Example: 0.2",
    )
    parser.add_argument(
        "--random-state",
        dest="RANDOM_STATE",
        type=int,
        default=None,
        help="Random seed integer.",
    )
    parser.add_argument(
        "--n-estimators",
        dest="N_ESTIMATORS",
        type=int,
        default=None,
        help="Number of trees when ESTIMATOR_GRID is empty.",
    )
    parser.add_argument(
        "--max-depth",
        dest="MAX_DEPTH",
        type=int,
        default=None,
        help="Max depth for trees. Use 0 or negative to mean None.",
    )
    parser.add_argument(
        "--estimator-grid",
        dest="ESTIMATOR_GRID",
        type=str,
        default=None,
        help="Comma-separated list for n_estimators search, e.g. '50,100,200'. Empty string disables grid.",
    )
    parser.add_argument(
        "--model-out",
        dest="MODEL_OUT",
        type=str,
        default=None,
        help="Where to save the trained pipeline.",
    )
    parser.add_argument(
        "--verbose",
        dest="VERBOSE",
        type=int,
        default=None,
        help="0 for quiet, 1 for brief logging.",
    )

    _args = parser.parse_args()

    # Apply overrides into module globals so later cells see final values
    _g = globals()
    for _name in [
        "DATA_PATH",
        "TARGET_COLUMN",
        "TEST_SIZE",
        "RANDOM_STATE",
        "N_ESTIMATORS",
        "MAX_DEPTH",
        "ESTIMATOR_GRID",
        "MODEL_OUT",
        "VERBOSE",
    ]:
        _val = getattr(_args, _name)
        if _val is not None:
            if _name == "MAX_DEPTH" and isinstance(_val, int) and _val <= 0:
                _val = None
            _g[_name] = _val

# %%
# Data loading (Iris by default, or user CSV)
if DATA_PATH:
    DATA_PATH = str(Path(DATA_PATH))
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"CSV not found: {DATA_PATH}")
    if not TARGET_COLUMN:
        raise ValueError("TARGET_COLUMN must be set when DATA_PATH is provided.")
    df = pd.read_csv(DATA_PATH)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in CSV.")
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    task = (
        "classification" if y.nunique() < max(20, int(0.05 * len(y))) else "regression"
    )
    if VERBOSE:
        print(f"[data] Loaded CSV with shape {df.shape}. Task inferred as {task}.")
else:
    from sklearn.datasets import load_iris

    iris = load_iris(as_frame=True)
    df = iris.frame
    TARGET_COLUMN = "target"
    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])
    task = "classification"
    if VERBOSE:
        print(f"[data] Loaded Iris dataset with shape {df.shape}.")

# %%
# Train/validation split and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

stratify = y if task == "classification" else None
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
)

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="drop",
)

if VERBOSE:
    print(
        f"[split] X_train={X_train.shape}, X_test={X_test.shape}, "
        f"num_cols={len(num_cols)}, cat_cols={len(cat_cols)}"
    )

# %%
# Model setup (RandomForest for classification, RandomForestRegressor for regression)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

if task == "classification":
    BaseModel = RandomForestClassifier
    score_name = "accuracy"
else:
    BaseModel = RandomForestRegressor
    score_name = "r2"

# Parse estimator grid
grid = []
if isinstance(ESTIMATOR_GRID, str):
    if ESTIMATOR_GRID.strip():
        grid = [int(s) for s in ESTIMATOR_GRID.split(",") if s.strip()]
if not grid:
    grid = [int(N_ESTIMATORS)]

if VERBOSE:
    print(f"[model] Candidate n_estimators: {grid}")

# %%
# Training loop with a tiny manual search over n_estimators
best_score = -np.inf
best_n = None
best_pipe: Pipeline | None = None

for n in tqdm(grid, disable=(VERBOSE == 0), desc="Training"):
    model = BaseModel(
        n_estimators=int(n),
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
        n_jobs=-1 if hasattr(BaseModel(), "n_jobs") else None,
    )
    pipe = Pipeline(steps=[("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)

    score = pipe.score(X_test, y_test)
    if VERBOSE:
        print(f"[fit] n_estimators={n} -> {score_name}={score:.4f}")

    if score > best_score:
        best_score = score
        best_n = n
        best_pipe = pipe

if VERBOSE and best_pipe is not None:
    print(f"[best] n_estimators={best_n} with {score_name}={best_score:.4f}")

# %%
# Evaluation summary
if task == "classification":
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = best_pipe.predict(X_test)
    print("[eval] Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("[eval] Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
else:
    from sklearn.metrics import mean_absolute_error, r2_score

    y_pred = best_pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"[eval] MAE={mae:.4f}, R2={r2:.4f}")

# %%
# Save trained pipeline
if MODEL_OUT:
    from joblib import dump

    out_path = Path(MODEL_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump(best_pipe, out_path)
    if VERBOSE:
        print(f"[save] Saved model to {out_path.resolve()}")

# %%
# Quick interactive tryout (edit 'sample' to match your data columns)
# For Iris, sample is one row in the same column order as X.
if ipython is not None:
    try:
        sample = X_test.iloc[[0]]
        pred = best_pipe.predict(sample)
        print("[demo] One-sample prediction:", pred)
    except Exception as e:
        print(f"[demo] Skipped demo due to error: {e}")
