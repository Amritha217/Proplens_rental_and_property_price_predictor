"""
src/price/data_pipeline.py
----------------------------
Cleans the raw Dubai Land Department (DLD) transaction export and prepares
it for price-prediction modeling.

Two critical fixes baked in here (found by inspecting the raw sample):

1. trans_group_en mixes "Sales", "Mortgages", and other transaction types
   in the SAME `actual_worth` column. A mortgage amount is not a sale
   price -- it's a different quantity (loan value) that happens to share
   a column name. Training on unfiltered data means the target itself is
   inconsistent, which plausibly explains predictions running high in
   earlier attempts. We filter to Sales only.

2. `meter_sale_price` = actual_worth / procedure_area, computed by DLD
   itself (verified: 1,000,000 / 1123.05 = 890.43, an exact match in the
   raw sample). This is direct target leakage if used as a feature -- the
   same failure mode as the price_to_rent_ratio bug in the rent module,
   just baked into the raw government export this time. Dropped entirely.

Usage:
    python src/price/data_pipeline.py \
        --inputs data/raw/dld_transactions_part1.csv data/raw/dld_transactions_part2.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

TARGET = "price"

# Raw column -> clean column name. Arabic-labeled duplicates dropped;
# _en columns kept and renamed.
RENAME_MAP = {
    "actual_worth": "price",
    "area_name_en": "area_name",
    "building_name_en": "building_name",
    "has_parking": "has_parking",
    "instance_date": "transaction_date",
    "master_project_en": "master_project",
    "nearest_landmark_en": "nearest_landmark",
    "nearest_mall_en": "nearest_mall",
    "nearest_metro_en": "nearest_metro",
    "procedure_area": "area_sqm",
    "project_name_en": "project_name",
    "property_sub_type_en": "property_sub_type",
    "property_type_en": "property_type",
    "property_usage_en": "property_usage",
    "reg_type_en": "reg_type",       # Existing Properties vs Off-Plan Properties
    "rooms_en": "rooms",
    "trans_group_en": "trans_group", # Sales / Mortgages / Gifts
    "transaction_id": "transaction_id",
}

# Columns we never want as model features: Arabic duplicates, IDs with no
# predictive meaning, and (critically) the two leakage-risk columns.
DROP_ALWAYS = [
    "meter_sale_price",   # direct leakage: price / area, computed by DLD itself
    "meter_rent_price", "rent_value",  # rental-transaction fields, irrelevant to sale price
    "load_timestamp",
    "no_of_parties_role_1", "no_of_parties_role_2", "no_of_parties_role_3",  # party counts, not price-relevant
    # Redundant numeric ID columns -- we already keep the matching _en text
    # column (property_sub_type, property_type, reg_type, trans_group), so
    # these IDs would just be collinear duplicates of features we already have
    "property_sub_type_id", "property_type_id", "reg_type_id", "trans_group_id",
    "procedure_id", "procedure_name",  # granular procedure detail, trans_group already captures Sales
    "area_id",  # 1:1 with area_name -- keep the readable name, drop the numeric duplicate
    # High-cardinality free-text names: hundreds/thousands of distinct
    # buildings/projects/landmarks would blow up one-hot dimensionality
    # for little gain. We already extract near_landmark/near_mall/near_metro
    # as presence flags below -- that captures most of the useful signal
    # without the cardinality explosion.
    "building_name", "project_name", "master_project",
    "project_number",
    # NOTE: transaction_id and nearest_landmark/mall/metro are NOT dropped
    # here -- transaction_id is needed for dedup first, and the nearest_*
    # columns are needed to compute near_landmark/near_mall/near_metro
    # presence flags in engineer_features(). Both get dropped later, after
    # they've actually been used.
]


def load_raw(paths) -> pd.DataFrame:
    frames = [pd.read_csv(p, encoding="utf-8") for p in paths]
    df = pd.concat(frames, ignore_index=True)
    # Drop Arabic-labeled columns (keep only _en + the handful of
    # non-language columns), then rename the rest.
    ar_cols = [c for c in df.columns if c.endswith("_ar")]
    df = df.drop(columns=ar_cols, errors="ignore")
    df = df.rename(columns=RENAME_MAP)
    return df


def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- FIX 1: Sales only -- drop Mortgages, Gifts, etc. ---
    before = len(df)
    df = df[df["trans_group"] == "Sales"]
    print(f"Filtered to Sales only: {before} -> {len(df)} rows "
          f"(dropped {before - len(df)} non-sale transactions, e.g. mortgages)")

    # --- Residential units only, matching the filtering approach that
    #     worked in the original project (villas/land mixed with units
    #     was a known source of error) ---
    before = len(df)
    df = df[df["property_usage"] == "Residential"]
    df = df[df["property_type"] == "Unit"]
    print(f"Filtered to Residential Units only: {before} -> {len(df)} rows")

    # Drop duplicate transaction records if both files overlap -- do this
    # BEFORE dropping transaction_id below, since dedup needs it
    if "transaction_id" in df.columns:
        before_dedup = len(df)
        df = df.drop_duplicates(subset=["transaction_id"])
        print(f"Deduplicated by transaction_id: {before_dedup} -> {len(df)} rows")

    # --- Drop leakage / irrelevant columns ---
    df = df.drop(columns=[c for c in DROP_ALWAYS if c in df.columns], errors="ignore")
    df = df.drop(columns=["transaction_id"], errors="ignore")  # done using it for dedup

    # --- Basic sanity filtering on price & area ---
    df = df[df["price"].notna() & (df["price"] > 0)]
    df = df[df["area_sqm"].notna() & (df["area_sqm"] > 0)]

    # Explicit size sanity bounds. The price/sqm RATIO filter below misses
    # rows where BOTH price and area are tiny (e.g. area_sqm=1.0 with a
    # proportionally small price) -- the ratio looks normal even though
    # the absolute size is physically impossible for a residential unit.
    # 15 sqm is a generous floor (smaller than any real studio); 1500 sqm
    # a generous ceiling for a "Unit" (excludes obvious data errors while
    # still allowing large penthouses).
    before = len(df)
    df = df[(df["area_sqm"] >= 15) & (df["area_sqm"] <= 1500)]
    print(f"Filtered implausible unit sizes (<15 or >1500 sqm): {before} -> {len(df)} rows")

    # Recompute price/sqm ourselves post-filter (not the leaked DLD
    # column) purely for outlier inspection, not as a model feature
    df["_price_per_sqm_check"] = df["price"] / df["area_sqm"]

    # Remove extreme outliers using percentile clipping on price/sqm
    # rather than raw price, since raw price naturally spans studios to
    # mansions -- price/sqm is the more meaningful outlier signal
    low, high = df["_price_per_sqm_check"].quantile([0.005, 0.995])
    before = len(df)
    df = df[(df["_price_per_sqm_check"] >= low) & (df["_price_per_sqm_check"] <= high)]
    print(f"Trimmed extreme price/sqm outliers (0.5%-99.5% band): {before} -> {len(df)} rows")
    df = df.drop(columns=["_price_per_sqm_check"])

    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Date features: confirmed format is ISO YYYY-MM-DD in the actual
    #     export (not DD-MM-YYYY -- that was a display artifact, not the
    #     real raw format). Keeping this simple per your request: mainly
    #     the year the transaction happened at that price, since that's
    #     what matters most for "what year was the property worth this."
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], format="%Y-%m-%d", errors="coerce")
    before = len(df)
    df = df[df["transaction_date"].notna()]
    print(f"Date parsing: {before} -> {len(df)} rows (dropped rows with unparseable dates)")
    df["year"] = df["transaction_date"].dt.year
    df["month"] = df["transaction_date"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["t_index"] = (df["year"] - df["year"].min()) * 12 + df["month"]

    # --- Amenity / context flags (mined from presence, not identity, to
    #     avoid extreme cardinality from hundreds of distinct landmark names) ---
    df["has_parking"] = df["has_parking"].fillna(0).astype(int)
    df["near_landmark"] = df["nearest_landmark"].notna().astype(int)
    df["near_mall"] = df["nearest_mall"].notna().astype(int)
    df["near_metro"] = df["nearest_metro"].notna().astype(int)
    df["is_offplan"] = (df["reg_type"] == "Off-Plan Properties").astype(int)

    # Now safe to drop the raw high-cardinality text columns -- their
    # useful signal (presence) is captured in the flags above already
    df = df.drop(columns=["nearest_landmark", "nearest_mall", "nearest_metro"], errors="ignore")

    # --- Rooms: parse "1 B/R", "Studio", etc. into a numeric bedroom count ---
    def parse_rooms(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip().lower()
        if "studio" in val:
            return 0
        digits = "".join(c for c in val if c.isdigit())
        return int(digits) if digits else np.nan

    df["bedrooms"] = df["rooms"].apply(parse_rooms)

    # --- Area name: leakage-safe (out-of-fold) target encoding done at
    #     train time in train.py, not here -- doing it here on the full
    #     dataset before any split would itself be a leak. This function
    #     only prepares the raw categorical column for that later step.
    df["area_name"] = df["area_name"].fillna("Unknown").str.strip()
    df["property_sub_type"] = df["property_sub_type"].fillna("Unknown").str.strip()

    return df


NUMERIC_FEATURES = [
    "area_sqm", "bedrooms", "has_parking", "near_landmark", "near_mall",
    "near_metro", "is_offplan", "year", "month_sin", "month_cos", "t_index",
]
ONEHOT_CATEGORICAL = ["property_sub_type"]
TARGET_ENCODE_CATEGORICAL = ["area_name"]  # high-cardinality -> target encoding, not one-hot


def time_split(df, test_start_year=2025):
    train = df[df["year"] < test_start_year].copy()
    test = df[df["year"] >= test_start_year].copy()
    return train, test


def run(input_paths, output_dir="data/processed_price", test_start_year=2025):
    os.makedirs(output_dir, exist_ok=True)

    df = load_raw(input_paths)
    print(f"Loaded raw: {len(df)} rows total (both files combined)")

    df = filter_and_clean(df)
    df = engineer_features(df)
    base_year = int(df["year"].min())  # t_index inside engineer_features is relative to this

    before = len(df)
    df = df[df["year"] >= 2007]
    print(f"Dropped pre-2007 rows (only 19 rows across 2003-2006, essentially noise): {before} -> {len(df)}")

    train, test = time_split(df, test_start_year)
    print(f"Final: {len(df)} rows | train: {len(train)} | test: {len(test)}")

    train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    df.to_csv(os.path.join(output_dir, "full_processed.csv"), index=False)

    spec = {
        "numeric": NUMERIC_FEATURES,
        "onehot_categorical": ONEHOT_CATEGORICAL,
        "target_encode_categorical": TARGET_ENCODE_CATEGORICAL,
        "target": TARGET,
        "base_year": base_year,  # t_index = (year - base_year) * 12 + month; needed to score new requests consistently
    }
    with open(os.path.join(output_dir, "feature_columns.json"), "w") as f:
        json.dump(spec, f, indent=2)

    print(f"\nOverall price distribution after cleaning:\n{df['price'].describe()}")

    df["_price_per_sqm"] = df["price"] / df["area_sqm"]
    area_stats = (
        df.groupby("area_name")["_price_per_sqm"]
        .agg(["mean", "median", "count"])
        .sort_values("mean")
    )
    print(f"\nPrice/sqm by area -- cheapest 10:\n{area_stats.head(10)}")
    print(f"\nPrice/sqm by area -- most expensive 10:\n{area_stats.tail(10)}")
    print(f"\nNumber of distinct areas: {df['area_name'].nunique()}")
    df = df.drop(columns=["_price_per_sqm"])

    return train, test, spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_dir", default="data/processed_price")
    parser.add_argument("--test_start_year", type=int, default=2025)
    args = parser.parse_args()
    run(args.inputs, args.output_dir, args.test_start_year)