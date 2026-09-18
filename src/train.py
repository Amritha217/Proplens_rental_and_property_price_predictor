"""
train.py
--------
Trains 5 candidate models on the rent-prediction task and evaluates all of
them on the SAME time-based holdout, both overall and per-community (this
is where your previous project's price model silently failed -- good
global R2, terrible on specific segments -- so we check for that here too).

Usage:
    python src/train.py --data_dir data/processed --model_dir models
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor


def load_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    with open(os.path.join(data_dir, "feature_columns.json")) as f:
        spec = json.load(f)
    return train, test, spec


def build_preprocessor(numeric_cols, categorical_cols):
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )


def get_models(numeric_cols, categorical_cols):
    """Returns a dict of name -> sklearn-compatible pipeline/model.

    Linear/RF use a preprocessing pipeline (need scaling + one-hot).
    XGBoost/LightGBM also go through the pipeline for simplicity.
    CatBoost handles categoricals natively, so it gets raw columns instead
    (this avoids the one-hot column-mismatch problem you hit last time).
    """
    pre = build_preprocessor(numeric_cols, categorical_cols)

    models = {
        "linear": Pipeline([("pre", pre), ("model", LinearRegression())]),
        "random_forest": Pipeline([
            ("pre", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", RandomForestRegressor(
                n_estimators=400, max_depth=12, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            )),
        ]),
        "xgboost": Pipeline([
            ("pre", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", xgb.XGBRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
            )),
        ]),
        "lightgbm": Pipeline([
            ("pre", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", lgb.LGBMRegressor(
                n_estimators=500, max_depth=-1, learning_rate=0.05,
                num_leaves=31, random_state=42,
            )),
        ]),
        # CatBoost trained separately below (native categorical handling)
    }
    return models


def evaluate(y_true, y_pred):
    return {
        "r2": round(r2_score(y_true, y_pred), 4),
        "mae": round(mean_absolute_error(y_true, y_pred), 3),
        "mape": round(mean_absolute_percentage_error(y_true, y_pred) * 100, 2),
    }


def per_community_error(test_df, y_pred, community_col="community_key", target_col="rental_price_per_sqft_annual_usd"):
    tmp = test_df[[community_col, target_col]].copy()
    tmp["pred"] = y_pred
    tmp["abs_pct_err"] = (tmp["pred"] - tmp[target_col]).abs() / tmp[target_col]
    return (
        tmp.groupby(community_col)["abs_pct_err"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"abs_pct_err": "mean_abs_pct_error"})
    )


def run(data_dir="data/processed", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    train, test, spec = load_data(data_dir)

    numeric_cols = spec["numeric"]
    categorical_cols = spec["categorical"]
    target = spec["target"]

    # Drop rows with NaN in lag features -- first months of each community
    # won't have lag_12 etc. This is expected and fine; just don't train on it.
    train_clean = train.dropna(subset=numeric_cols + [target])
    test_clean = test.dropna(subset=numeric_cols + [target])

    X_train, y_train = train_clean[numeric_cols + categorical_cols], train_clean[target]
    X_test, y_test = test_clean[numeric_cols + categorical_cols], test_clean[target]

    results = {}
    models = get_models(numeric_cols, categorical_cols)

    for name, pipe in models.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        metrics = evaluate(y_test, preds)
        results[name] = metrics
        joblib.dump(pipe, os.path.join(model_dir, f"{name}_model.pkl"))
        print(f"  {name}: {metrics}")

        worst = per_community_error(test_clean, preds).head(5)
        if not worst.empty and worst["mean_abs_pct_error"].iloc[0] > 0.25:
            print(f"  WARNING: {name} has >25% avg error on some communities:")
            print(worst.to_string(index=False))

    # --- CatBoost separately: native categorical support ---
    print("Training catboost...")
    cb_model = CatBoostRegressor(
        iterations=600, depth=8, learning_rate=0.05,
        loss_function="RMSE", cat_features=categorical_cols,
        verbose=False, random_state=42,
    )
    cb_model.fit(X_train, y_train)
    cb_preds = cb_model.predict(X_test)
    results["catboost"] = evaluate(y_test, cb_preds)
    cb_model.save_model(os.path.join(model_dir, "catboost_model.cbm"))
    print(f"  catboost: {results['catboost']}")

    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    best_model = min(results, key=lambda k: results[k]["mape"])
    print(f"\nBest model by MAPE: {best_model} -> {results[best_model]}")
    with open(os.path.join(model_dir, "best_model.json"), "w") as f:
        json.dump({"best_model": best_model, "metrics": results[best_model]}, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()
    run(args.data_dir, args.model_dir)