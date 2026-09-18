"""
src/price/train.py
--------------------
Trains and benchmarks 5 models (Linear, RF, XGBoost, LightGBM, CatBoost)
on the cleaned DLD sales data.

Two things done carefully here, both directly tied to problems from the
original (failed) price project:

1. LEAKAGE-SAFE target encoding for `area_name`. Naively encoding each
   area by its mean price computed from the FULL dataset (including the
   row being predicted) leaks the target. This uses out-of-fold encoding
   on the train set (a row's encoding is computed only from OTHER folds)
   and a train-only global encoding applied to the test set.

2. Per-price-tier error diagnostic on the single full-data model, BEFORE
   deciding whether segment-specific models are needed. Your original
   project discovered the single-model failure the hard way, after
   deployment. Here we check for it proactively: if the single model's
   error blows up on budget/luxury tiers, segment-specific models are
   trained as the fix -- same solution that worked before, but arrived at
   by diagnosis rather than trial and error.

Usage:
    python src/price/train.py --data_dir data/processed_price --model_dir models/price
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error, mean_absolute_percentage_error,
    mean_squared_error, r2_score,
)
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


# ---------------- Leakage-safe target encoding ----------------

def out_of_fold_target_encode(train_df, col, target_col, n_splits=5, smoothing=20, seed=42):
    """Returns (encoded_train_series, encoding_map, global_mean).

    For each row in the train set, its encoded value is the smoothed mean
    target of `col` computed from the OTHER folds only -- never including
    the row itself or its fold. This is what makes it leakage-safe: a
    naive `df.groupby(col)[target].transform('mean')` would include each
    row's own price in its own area's average, which is a real, if subtle,
    leak (and inflates apparent accuracy the same way the rent module's
    price_to_rent_ratio bug did).

    Smoothing blends each area's own mean with the global mean, weighted
    by how many training rows that area has -- this keeps rare areas from
    getting a noisy, overconfident encoding from just a few rows.
    """
    global_mean = train_df[target_col].mean()
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    encoded = pd.Series(index=train_df.index, dtype=float)

    for train_idx, holdout_idx in kf.split(train_df):
        fold_train = train_df.iloc[train_idx]
        stats = fold_train.groupby(col)[target_col].agg(["mean", "count"])
        smoothed = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        encoded.iloc[holdout_idx] = train_df.iloc[holdout_idx][col].map(smoothed).values

    # Any area that appeared in a holdout fold but not enough in the
    # corresponding train folds falls back to the global mean
    encoded = encoded.fillna(global_mean)

    # Full-train encoding map (for applying to the TEST set, which sees
    # the whole train set as its "other folds" -- this is fine, test rows
    # are never used to compute it)
    full_stats = train_df.groupby(col)[target_col].agg(["mean", "count"])
    full_map = (full_stats["mean"] * full_stats["count"] + global_mean * smoothing) / (full_stats["count"] + smoothing)

    return encoded, full_map, global_mean


def apply_target_encoding(df, col, encoding_map, global_mean, new_col_name=None):
    new_col_name = new_col_name or f"{col}_target_enc"
    df = df.copy()
    df[new_col_name] = df[col].map(encoding_map).fillna(global_mean)
    return df


# ---------------- Model setup ----------------

def build_pipeline(numeric_cols, categorical_cols, model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    return Pipeline([("pre", pre), ("model", model)])


def get_models():
    return {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=16, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "xgboost": xgb.XGBRegressor(
            n_estimators=500, max_depth=7, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
        ),
        "lightgbm": lgb.LGBMRegressor(
            n_estimators=500, max_depth=-1, learning_rate=0.05,
            num_leaves=63, random_state=42,
        ),
    }


def evaluate(y_true, y_pred):
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 0),
        "rmse": round(mean_squared_error(y_true, y_pred) ** 0.5, 0),
        "mape": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
    }


def price_tier_error(test_df, y_pred, target_col, tier_edges):
    tmp = test_df[[target_col]].copy()
    tmp["pred"] = y_pred
    tmp["tier"] = pd.cut(tmp[target_col], bins=tier_edges, include_lowest=True)
    return tmp.groupby("tier", observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "mape": mean_absolute_percentage_error(g[target_col], g["pred"]) * 100,
        })
    )


# ---------------- Main run ----------------

def run(data_dir="data/processed_price", model_dir="models/price"):
    os.makedirs(model_dir, exist_ok=True)

    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    with open(os.path.join(data_dir, "feature_columns.json")) as f:
        spec = json.load(f)

    target = spec["target"]
    numeric_cols = list(spec["numeric"])
    onehot_cols = spec["onehot_categorical"]
    te_col = spec["target_encode_categorical"][0]  # "area_name"

    train = train.dropna(subset=numeric_cols + [target, te_col])
    test = test.dropna(subset=numeric_cols + [target, te_col])

    # --- Leakage-safe target encoding for area_name ---
    encoded_train_col, full_map, global_mean = out_of_fold_target_encode(train, te_col, target)
    train["area_target_enc"] = encoded_train_col
    test = apply_target_encoding(test, te_col, full_map, global_mean, "area_target_enc")

    numeric_cols_final = numeric_cols + ["area_target_enc"]

    with open(os.path.join(model_dir, "area_encoding_map.json"), "w") as f:
        json.dump({"map": full_map.to_dict(), "global_mean": global_mean}, f, indent=2)

    X_train, y_train = train[numeric_cols_final + onehot_cols], train[target]
    X_test, y_test = test[numeric_cols_final + onehot_cols], test[target]

    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}")

    # --- Train & benchmark 5 models on the FULL (non-segmented) data ---
    results = {}
    fitted_pipelines = {}

    for name, model in get_models().items():
        print(f"Training {name}...")
        pipe = build_pipeline(numeric_cols_final, onehot_cols, model)
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        results[name] = evaluate(y_test, preds)
        fitted_pipelines[name] = pipe
        joblib.dump(pipe, os.path.join(model_dir, f"{name}_full_model.pkl"))
        print(f"  {name}: {results[name]}")

    print("Training catboost...")
    cb = CatBoostRegressor(
        iterations=600, depth=9, learning_rate=0.05, loss_function="RMSE",
        cat_features=onehot_cols, verbose=False, random_state=42,
    )
    cb.fit(X_train, y_train)
    cb_preds = cb.predict(X_test)
    results["catboost"] = evaluate(y_test, cb_preds)
    cb.save_model(os.path.join(model_dir, "catboost_full_model.cbm"))
    print(f"  catboost: {results['catboost']}")

    with open(os.path.join(model_dir, "full_model_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    best_name = min(results, key=lambda k: results[k]["mape"])
    print(f"\nBest single model by MAPE: {best_name} -> {results[best_name]}")

    # --- Diagnostic: per price-tier error on the best single model ---
    best_preds = cb_preds if best_name == "catboost" else fitted_pipelines[best_name].predict(X_test)
    tier_edges = list(train[target].quantile([0, 0.33, 0.66, 1.0]))
    tier_report = price_tier_error(test, best_preds, target, tier_edges)
    print(f"\nPer price-tier error ({best_name}):\n{tier_report}")

    worst_tier_mape = tier_report["mape"].max()
    needs_segmentation = worst_tier_mape > 1.5 * results[best_name]["mape"]

    with open(os.path.join(model_dir, "best_model.json"), "w") as f:
        json.dump({
            "best_model": best_name,
            "metrics": results[best_name],
            "tier_edges": tier_edges,
            "worst_tier_mape": round(float(worst_tier_mape), 2),
            "needs_segmentation": bool(needs_segmentation),
        }, f, indent=2)

    if needs_segmentation:
        print(f"\nWARNING: worst price tier's MAPE ({worst_tier_mape:.1f}%) is much higher than "
              f"the overall MAPE ({results[best_name]['mape']}%). This is the same failure pattern "
              f"as the original project -- run train_segments.py next to fix it with "
              f"segment-specific models.")
    else:
        print(f"\nPer-tier error looks reasonably consistent -- segmentation may not be "
              f"necessary this time. Still worth spot-checking a few real properties "
              f"manually before trusting this.")

    return results, tier_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed_price")
    parser.add_argument("--model_dir", default="models/price")
    args = parser.parse_args()
    run(args.data_dir, args.model_dir)