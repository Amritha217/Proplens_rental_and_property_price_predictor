"""
src/price/spot_check.py
-------------------------
Quick manual sanity check on individual predictions, WITH calibration
applied -- same discipline as before: never trust an aggregate metric
without eyeballing real rows.

Usage:
    python src/price/spot_check.py --data_dir data/processed_price --model_dir models/price --n 8
"""

import argparse
import json
import os

import joblib
import pandas as pd

from calibrate import apply_calibration


def run(data_dir="data/processed_price", model_dir="models/price", n=8, seed=7):
    test = pd.read_csv(os.path.join(data_dir, "test.csv"))

    with open(os.path.join(model_dir, "best_model.json")) as f:
        best = json.load(f)
    model = joblib.load(os.path.join(model_dir, f"{best['best_model']}_full_model.pkl"))
    feature_cols = list(model.named_steps["pre"].feature_names_in_)

    with open(os.path.join(model_dir, "area_encoding_map.json")) as f:
        enc = json.load(f)
    with open(os.path.join(model_dir, "trend_calibration.json")) as f:
        calib = json.load(f)

    sample = test.sample(n, random_state=seed).copy()
    sample["area_target_enc"] = sample["area_name"].map(enc["map"]).fillna(enc["global_mean"])

    raw_preds = model.predict(sample[feature_cols])
    calibrated_preds = apply_calibration(raw_preds, sample["t_index"], calib)

    sample["raw_pred"] = raw_preds.round(0)
    sample["calibrated_pred"] = calibrated_preds.round(0)
    sample["error_pct"] = ((sample["calibrated_pred"] - sample["price"]) / sample["price"] * 100).round(1)

    cols = ["area_name", "property_sub_type", "area_sqm", "bedrooms", "year",
            "price", "raw_pred", "calibrated_pred", "error_pct"]
    print(sample[cols].to_string(index=False))
    print(f"\nMean abs error: {sample['error_pct'].abs().mean():.1f}%")
    print(f"Worst single error: {sample['error_pct'].abs().max():.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed_price")
    parser.add_argument("--model_dir", default="models/price")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    run(args.data_dir, args.model_dir, args.n, args.seed)