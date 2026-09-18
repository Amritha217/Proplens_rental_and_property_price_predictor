"""
src/price/calibrate.py
------------------------
Fixes the systematic underprediction confirmed for 2025-2026: tree models
(LightGBM/RF/XGBoost/CatBoost) cannot extrapolate a trend past the range
of values they were trained on. Training stops at 2024, so the model has
no way to know 2025/2026 prices kept climbing -- it just predicts near
2024 price levels for any later year, producing a bias that gets worse
the further past 2024 you go (confirmed: -4.2% mean in 2025, -7.0% in
2026).

Fix: fit a simple market trend (log-linear price/sqm growth vs. time)
from RECENT training data only, then multiply out-of-range predictions
by the extrapolated growth factor. This is a standard, explainable
post-processing correction for this exact limitation -- not a change to
the model itself.

Usage:
    python src/price/calibrate.py --data_dir data/processed_price --model_dir models/price
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error


def compute_trend_calibration(train_df, target_col="price", area_col="area_sqm",
                               t_col="t_index", recent_months=24):
    """Fit log-linear price/sqm growth using only the most recent N months
    of training data (recent trend, not skewed by e.g. the 2009 crash or
    2014 peak sitting years earlier in the same training set)."""
    df = train_df.copy()
    df["price_per_sqm"] = df[target_col] / df[area_col]
    max_t = df[t_col].max()

    recent = df[df[t_col] >= max_t - recent_months]
    monthly = recent.groupby(t_col)["price_per_sqm"].median().reset_index()

    X = monthly[[t_col]].values
    y = np.log(monthly["price_per_sqm"].values)
    reg = LinearRegression().fit(X, y)
    monthly_log_growth = float(reg.coef_[0])

    return {
        "monthly_log_growth": monthly_log_growth,
        "max_train_t_index": int(max_t),
        "annualized_growth_pct": round((np.exp(monthly_log_growth * 12) - 1) * 100, 2),
    }


def apply_calibration(preds, t_index_series, calib):
    """Only adjusts rows PAST the training range -- rows within the
    training range are left untouched, since the model already learned
    those directly and doesn't need extrapolation help."""
    growth = calib["monthly_log_growth"]
    max_t = calib["max_train_t_index"]
    months_ahead = np.clip(t_index_series.values - max_t, a_min=0, a_max=None)
    factor = np.exp(growth * months_ahead)
    return np.asarray(preds) * factor


def run(data_dir="data/processed_price", model_dir="models/price"):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    with open(os.path.join(model_dir, "best_model.json")) as f:
        best = json.load(f)
    best_name = best["best_model"]

    with open(os.path.join(model_dir, "area_encoding_map.json")) as f:
        enc = json.load(f)

    if best_name == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(os.path.join(model_dir, "catboost_full_model.cbm"))
        feature_cols = None  # CatBoost pipeline stored differently; see note below
    else:
        model = joblib.load(os.path.join(model_dir, f"{best_name}_full_model.pkl"))
        feature_cols = list(model.named_steps["pre"].feature_names_in_)

    test["area_target_enc"] = test["area_name"].map(enc["map"]).fillna(enc["global_mean"])
    raw_preds = model.predict(test[feature_cols])

    calib = compute_trend_calibration(train)
    print(f"Trend calibration: {calib['annualized_growth_pct']}% annualized growth "
          f"(fit from last 24 months of training data)")

    calibrated_preds = apply_calibration(raw_preds, test["t_index"], calib)

    print("\n=== Before calibration ===")
    before = test.groupby("year").apply(
        lambda g: mean_absolute_percentage_error(g["price"], raw_preds[g.index]) * 100
    )
    print(before)
    before_bias = test.groupby("year").apply(
        lambda g: ((raw_preds[g.index] - g["price"]) / g["price"] * 100).mean()
    )
    print("Mean bias (before):\n", before_bias)

    print("\n=== After calibration ===")
    after = test.groupby("year").apply(
        lambda g: mean_absolute_percentage_error(g["price"], calibrated_preds[g.index]) * 100
    )
    print(after)
    after_bias = test.groupby("year").apply(
        lambda g: ((calibrated_preds[g.index] - g["price"]) / g["price"] * 100).mean()
    )
    print("Mean bias (after):\n", after_bias)

    with open(os.path.join(model_dir, "trend_calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nSaved calibration params to {model_dir}/trend_calibration.json "
          f"(used at inference time in the API)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed_price")
    parser.add_argument("--model_dir", default="models/price")
    args = parser.parse_args()
    run(args.data_dir, args.model_dir)