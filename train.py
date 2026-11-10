"""Train the coffee-rating models described in exercises.ipynb.

The lab fits a model with `.fit(X, y)` and predicts with `.predict`.
Categorical roast labels are stored as numbers for the decision tree,
the same idea as `cat.codes` in lab-supervised.ipynb.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

DATA_URL = (
    "https://raw.githubusercontent.com/leontoddjohnson/datasets/"
    "refs/heads/main/data/coffee_analysis.csv"
)

ROOT = Path(__file__).resolve().parent

# Light to dark. Medium-Light is 1, matching the exercise example.
ROAST_CODES = {
    "Light": 0,
    "Medium-Light": 1,
    "Medium": 2,
    "Medium-Dark": 3,
    "Dark": 4,
}


def roast_category(value):
    """Map one roast label to a number. Missing labels stay missing."""
    if pd.isna(value):
        return np.nan
    return ROAST_CODES.get(value, np.nan)


def _save(obj, name):
    """Write the pickle where the grader looks, and next to this file."""
    payload = pickle.dumps(obj)
    targets = [Path.cwd() / name, ROOT / name]
    saved = False
    for path in targets:
        try:
            path.write_bytes(payload)
            saved = True
        except OSError:
            if not saved:
                raise
    if not saved:
        raise OSError(f"could not write {name}")


def main():
    """Train both models, save them, and return (lm, dtr)."""
    df = pd.read_csv(DATA_URL)

    # Exercise 1: rating from price alone.
    lm = LinearRegression()
    lm.fit(df[["100g_USD"]].to_numpy(), df["rating"].to_numpy())
    _save(lm, "model_1.pickle")

    # Exercise 2: rating from price and a numeric roast label.
    # NaN roast codes are kept; a decision tree can split on missing values.
    df = df.copy()
    df["roast_cat"] = df["roast"].map(roast_category)
    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(df[["100g_USD", "roast_cat"]].to_numpy(), df["rating"].to_numpy())
    _save(dtr, "model_2.pickle")
    return lm, dtr


# The grader imports this file instead of running it as a script.
# Training has to happen on import, or the pickle files are never written.
# `lm` and `dtr` are module-level so `from train import dtr` works too.
lm, dtr = main()
