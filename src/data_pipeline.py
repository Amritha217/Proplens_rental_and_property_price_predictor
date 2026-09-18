"""
data_pipeline.py
-----------------
Loads the raw community-month rental panel data, cleans it, engineers
features, and produces a time-based train/test split.

Grain of this data: one row per (community, year_month) -- NOT per
individual listing. This is critical: it means a random train/test split
would leak future months into training. We ALWAYS split by time.

Usage:
    python src/data_pipeline.py --input data/raw/dubai_rent.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "year_month", "community", "zone", "is_freehold",
    "secondary_price_per_sqft_usd", "secondary_price_per_m2_usd",
    "offplan_price_per_sqft_usd", "rental_price_per_sqft_annual_usd",
    "n_listings_secondary", "n_listings_offplan", "n_listings_rental",
    "cbuae_base_rate_pct", "avg_mortgage_rate_pct",
]

TARGET = "rental_price_per_sqft_annual_usd"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set(RAW_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse date
    df["year_month"] = pd.to_datetime(df["year_month"], format="%Y-%m")

    # Normalize text fields (this is where "Al Warsan" vs "Al Warsan First"
    # style mismatches happen -- strip/lower for matching, keep original for display)
    df["community"] = df["community"].astype(str).str.strip()
    df["zone"] = df["zone"].astype(str).str.strip()
    df["community_key"] = df["community"].str.lower().str.replace(r"\s+", " ", regex=True)

    # Boolean cleanup
    if df["is_freehold"].dtype == object:
        df["is_freehold"] = df["is_freehold"].astype(str).str.upper().map(
            {"TRUE": True, "FALSE": False}
        )

    # Drop rows with no target or non-positive rent/price (data errors, not real zeros)
    df = df[df[TARGET].notna() & (df[TARGET] > 0)]
    df = df[df["secondary_price_per_sqft_usd"] > 0]

    # Drop rows with very thin listing support -- these are noisy signals,
    # not necessarily errors, but they should not be treated with the same
    # confidence as well-supported rows. We keep them but flag them.
    df["low_confidence"] = df["n_listings_rental"] < 5

    df = df.sort_values(["community_key", "year_month"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Calendar features ---
    df["year"] = df["year_month"].dt.year
    df["month"] = df["year_month"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    # Months since start of dataset -- captures long-run trend
    df["t_index"] = (
        (df["year_month"].dt.year - df["year_month"].dt.year.min()) * 12
        + df["year_month"].dt.month
    )

    # --- Lag / rolling features, computed PER COMMUNITY ---
    # These are usually the strongest predictors in panel/time-series data.
    g = df.groupby("community_key")[TARGET]
    df["rent_lag_1"] = g.shift(1)
    df["rent_lag_3"] = g.shift(3)
    df["rent_lag_12"] = g.shift(12)
    df["rent_roll_mean_3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    df["rent_roll_mean_12"] = g.transform(lambda s: s.shift(1).rolling(12).mean())

    # --- Cross-signal features ---
    # IMPORTANT: do NOT divide by df[TARGET] here -- that leaks the answer
    # directly into a feature (price / current_rent), which is why the
    # first training run scored a suspicious R2=1.0. Use the *lagged* rent
    # instead, which is legitimately known before the row we're predicting.
    df["price_to_rent_ratio_lag1"] = (
        df["secondary_price_per_sqft_usd"] / df["rent_lag_1"]
    )
    df["offplan_secondary_price_gap"] = (
        df["offplan_price_per_sqft_usd"] - df["secondary_price_per_sqft_usd"]
    )
    df["total_listings"] = (
        df["n_listings_secondary"] + df["n_listings_offplan"] + df["n_listings_rental"]
    )
    # Rental demand share -- how much of all market activity is rental vs sale
    df["rental_activity_share"] = df["n_listings_rental"] / df["total_listings"].replace(0, np.nan)

    # --- Macro features already present: cbuae_base_rate_pct, avg_mortgage_rate_pct ---
    # keep as-is, they're already numeric and monthly.

    return df


def time_split(df: pd.DataFrame, test_start: str = "2025-01"):
    """Split by calendar time, never randomly, to avoid leaking the future
    into training (this is the mistake that inflated the old price model's
    test-set R2 without it actually generalizing)."""
    cutoff = pd.to_datetime(test_start, format="%Y-%m")
    train = df[df["year_month"] < cutoff].copy()
    test = df[df["year_month"] >= cutoff].copy()
    return train, test


FEATURE_COLUMNS_NUMERIC = [
    "secondary_price_per_sqft_usd", "secondary_price_per_m2_usd",
    "offplan_price_per_sqft_usd", "n_listings_secondary",
    "n_listings_offplan", "n_listings_rental", "cbuae_base_rate_pct",
    "avg_mortgage_rate_pct", "year", "month_sin", "month_cos", "t_index",
    "rent_lag_1", "rent_lag_3", "rent_lag_12", "rent_roll_mean_3",
    "rent_roll_mean_12", "price_to_rent_ratio_lag1", "offplan_secondary_price_gap",
    "total_listings", "rental_activity_share",
]
FEATURE_COLUMNS_CATEGORICAL = ["community_key", "zone", "is_freehold"]


def run(input_path: str, output_dir: str = "data/processed", test_start: str = "2025-01"):
    os.makedirs(output_dir, exist_ok=True)

    df = load_raw(input_path)
    df = clean(df)
    df = engineer_features(df)

    train, test = time_split(df, test_start=test_start)

    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    df.to_csv(os.path.join(output_dir, "full_processed.csv"), index=False)

    feature_spec = {
        "numeric": FEATURE_COLUMNS_NUMERIC,
        "categorical": FEATURE_COLUMNS_CATEGORICAL,
        "target": TARGET,
        "test_start": test_start,
    }
    with open(os.path.join(output_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_spec, f, indent=2)

    print(f"Rows total: {len(df)} | train: {len(train)} | test: {len(test)}")
    print(f"Saved processed data + feature spec to {output_dir}/")
    return train, test, feature_spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to raw rent CSV")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--test_start", default="2025-01")
    args = parser.parse_args()
    run(args.input, args.output_dir, args.test_start)