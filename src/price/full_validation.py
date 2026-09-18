"""
src/price/full_validation.py
------------------------------
Runs predictions across the FULL test set (not just a handful of manual
spot-checks) and breaks down error by the confidence flag from explain.py.

The point: if "low confidence" predictions really do have much worse
error than "high confidence" ones, that proves the confidence flag is
doing real work, not just decorative. If error looks the same across all
three tiers, the flag isn't actually catching what it claims to.

Usage:
    python src/price/full_validation.py --data_dir ../../data/processed_price --model_dir ../../models/price
"""

import argparse
import os

import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

from explain import PriceExplainer


def run(data_dir="data/processed_price", model_dir="models/price"):
    explainer = PriceExplainer(model_dir=model_dir, data_dir=data_dir)
    test = pd.read_csv(os.path.join(data_dir, "test.csv")).dropna(
        subset=explainer.feature_cols_raw_needed() + ["price"]
    )

    print(f"Validating across full test set: {len(test)} rows")
    raw_preds, calibrated_preds, confidence = explainer.predict_batch(test)

    test = test.copy()
    test["predicted_price"] = calibrated_preds
    test["confidence"] = confidence.values
    test["abs_pct_error"] = ((test["predicted_price"] - test["price"]) / test["price"]).abs() * 100

    print("\n=== Overall ===")
    print(f"MAPE: {mean_absolute_percentage_error(test['price'], test['predicted_price']) * 100:.2f}%")

    print("\n=== By confidence tier ===")
    tier_summary = test.groupby("confidence").agg(
        n=("price", "count"),
        mape=("abs_pct_error", "mean"),
        median_error=("abs_pct_error", "median"),
        pct_over_30=("abs_pct_error", lambda s: (s > 30).mean() * 100),
    ).sort_values("mape")
    print(tier_summary)

    print("\n=== Worst 10 individual predictions (any confidence) ===")
    worst = test.sort_values("abs_pct_error", ascending=False).head(10)
    print(worst[["area_name", "property_sub_type", "area_sqm", "bedrooms",
                 "year", "price", "predicted_price", "confidence", "abs_pct_error"]])

    print("\n--- Interpretation ---")
    high_mape = tier_summary.loc["high", "mape"] if "high" in tier_summary.index else None
    low_mape = tier_summary.loc["low", "mape"] if "low" in tier_summary.index else None
    if high_mape is not None and low_mape is not None:
        if low_mape > high_mape * 1.3:
            print(f"Confidence flag is working: 'low' confidence MAPE ({low_mape:.1f}%) is "
                  f"meaningfully worse than 'high' confidence MAPE ({high_mape:.1f}%). "
                  f"The flag is correctly identifying less reliable predictions.")
        else:
            print(f"Confidence flag isn't clearly separating accuracy: 'low' MAPE ({low_mape:.1f}%) "
                  f"vs 'high' MAPE ({high_mape:.1f}%) aren't very different. May need better "
                  f"confidence criteria, but given the time budget this is worth noting as a "
                  f"limitation rather than fixing further right now.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed_price")
    parser.add_argument("--model_dir", default="models/price")
    args = parser.parse_args()
    run(args.data_dir, args.model_dir)