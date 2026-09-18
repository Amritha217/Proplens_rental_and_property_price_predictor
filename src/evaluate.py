"""
evaluate.py
-----------
Two checks that matter more than the headline MAPE:

1. Naive baseline comparison: how much better is the model than simply
   guessing "rent = last month's rent for this community"? If the model
   barely beats this, the model isn't adding real value -- the lag
   feature is doing all the work.

2. Cold-start error: error specifically on rows where the community has
   THIN history (few/no lag features available) -- this is exactly the
   scenario yield_engine.py falls back to the model for. Good average
   error elsewhere doesn't tell you anything about this case.

Usage:
    python src/evaluate.py --data_dir data/processed --model_dir models
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, r2_score


def naive_baseline(test_df, target_col="rental_price_per_sqft_annual_usd"):
    """Predict = last month's actual rent for that community (rent_lag_1)."""
    valid = test_df.dropna(subset=["rent_lag_1", target_col])
    mape = mean_absolute_percentage_error(valid[target_col], valid["rent_lag_1"]) * 100
    r2 = r2_score(valid[target_col], valid["rent_lag_1"])
    return {"r2": round(r2, 4), "mape": round(mape, 2), "n_rows": len(valid)}


def coldstart_error(test_df, model, numeric_cols, categorical_cols, target_col):
    """Rows with no rent_lag_12 (i.e. community has < 12 months of prior
    history in the dataset) approximate the 'thin comparables' case."""
    cold = test_df[test_df["rent_lag_12"].isna()].copy()
    if cold.empty:
        return {"n_rows": 0, "note": "No cold-start rows in this test set -- "
                "every community had 12+ months of history. Consider "
                "manually holding out a few communities entirely to test this."}

    # Model still needs its numeric features non-null; fill lag-derived
    # NaNs with the community's available mean as the model pipeline would
    # see at inference time in a genuine cold-start call.
    cold_filled = cold.copy()
    for col in numeric_cols:
        if cold_filled[col].isna().any():
            cold_filled[col] = cold_filled[col].fillna(cold_filled[col].mean())

    X_cold = cold_filled[numeric_cols + categorical_cols]
    y_cold = cold_filled[target_col]
    preds = model.predict(X_cold)

    mape = mean_absolute_percentage_error(y_cold, preds) * 100
    r2 = r2_score(y_cold, preds)
    return {"r2": round(r2, 4), "mape": round(mape, 2), "n_rows": len(cold)}


def community_holdout_eval(full_df, numeric_cols, categorical_cols, target,
                            n_holdout=6, seed=42):
    """True cold-start simulation: pick a handful of communities, remove
    ALL their rows from training (not just future months), then check
    error on their rows. This tells you how the model performs on a
    community it has genuinely never seen -- the real-world case for a
    brand-new/rare community in yield_engine.py's fallback path.

    This trains a throwaway model purely for this diagnostic -- it does
    NOT overwrite your production model files.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = full_df.dropna(subset=numeric_cols + [target]).copy()
    communities = df["community_key"].unique()
    rng = np.random.default_rng(seed)
    holdout_communities = rng.choice(communities, size=min(n_holdout, len(communities)), replace=False)

    train_mask = ~df["community_key"].isin(holdout_communities)
    train_df = df[train_mask]
    holdout_df = df[~train_mask].copy()

    # Also blank out the lag/rolling-derived features for the holdout rows.
    # A truly new/thin community has no rent history at all -- so it would
    # never have these values available either. Without this step, the
    # model is still handed the community's own recent rent as a feature,
    # which is the same shortcut the naive baseline uses, just unlabeled.
    lag_derived_cols = [c for c in numeric_cols if
                         c.startswith("rent_lag") or c.startswith("rent_roll") or
                         c == "price_to_rent_ratio_lag1"]
    global_means = train_df[lag_derived_cols].mean()
    holdout_df[lag_derived_cols] = global_means.values  # same values a brand-new community's row would get

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ])
    model = Pipeline([("pre", pre), ("model", RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1))])

    model.fit(train_df[numeric_cols + categorical_cols], train_df[target])
    preds = model.predict(holdout_df[numeric_cols + categorical_cols])

    mape = mean_absolute_percentage_error(holdout_df[target], preds) * 100
    r2 = r2_score(holdout_df[target], preds)
    return {
        "held_out_communities": list(holdout_communities),
        "r2": round(r2, 4),
        "mape": round(mape, 2),
        "n_rows": len(holdout_df),
    }


def run(data_dir="data/processed", model_dir="models"):
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))
    with open(os.path.join(data_dir, "feature_columns.json")) as f:
        spec = json.load(f)

    numeric_cols = spec["numeric"]
    categorical_cols = spec["categorical"]
    target = spec["target"]

    print("=== Naive baseline (predict = last month's rent) ===")
    baseline = naive_baseline(test, target)
    print(baseline)

    with open(os.path.join(model_dir, "best_model.json")) as f:
        best = json.load(f)
    best_name = best["best_model"]
    print(f"\n=== Cold-start check for best model: {best_name} ===")

    if best_name == "catboost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(os.path.join(model_dir, "catboost_model.cbm"))
    else:
        model = joblib.load(os.path.join(model_dir, f"{best_name}_model.pkl"))

    cold_result = coldstart_error(test, model, numeric_cols, categorical_cols, target)
    print(cold_result)

    print("\n=== True cold-start simulation (unseen communities) ===")
    full_df = pd.read_csv(os.path.join(data_dir, "full_processed.csv"))
    holdout_result = community_holdout_eval(full_df, numeric_cols, categorical_cols, target)
    print(holdout_result)

    print("\n--- Interpretation ---")
    if cold_result.get("n_rows", 0) == 0:
        print("Couldn't measure cold-start error directly -- see note above. "
              "Recommend re-running data_pipeline with a couple of communities "
              "fully held out of training to simulate a truly new community.")
    elif cold_result["mape"] > baseline["mape"] * 3:
        print("WARNING: cold-start error is much worse than the naive baseline's "
              "overall error. The model is not reliable for new/thin-history "
              "communities -- lean more heavily on market comparables in "
              "yield_engine.py for those cases (increase comp weight, lower "
              "trust/confidence flag).")
    else:
        print("Cold-start error is in a reasonable range relative to baseline. "
              "Model looks usable as a fallback for thin-comparable communities.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--model_dir", default="models")
    args = parser.parse_args()
    run(args.data_dir, args.model_dir)