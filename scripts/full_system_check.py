"""
scripts/full_system_check.py
------------------------------
One script to check diverse feature combinations across ALL THREE
prediction surfaces against real/actual values where actual values exist:

1. RENT: real community-month rows, predicted rent vs actual rent
2. PRICE: real DLD transaction rows, predicted price vs actual price
3. INVESTMENT SCORE: no ground truth exists for this (it's a derived
   metric, not an observed quantity) -- instead this prints the score
   breakdown across diverse communities so you can sanity-check the
   spread makes sense, same as the earlier manual testing in this
   conversation.

Run from project root:
    python scripts/full_system_check.py
"""

import json
import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "price"))

from src.yield_engine import PropertyInput, compute_yield_and_score, estimate_rent  # noqa: E402
from explain import PriceExplainer  # noqa: E402


def check_rent(n=8):
    print("\n" + "=" * 70)
    print("RENT: diverse communities/sizes, predicted vs actual")
    print("=" * 70)

    market_df = pd.read_csv("data/processed/full_processed.csv", parse_dates=["year_month"])
    with open("data/processed/feature_columns.json") as f:
        spec = json.load(f)
    model = joblib.load("models/random_forest_model.pkl")

    # Diverse feature combinations: different communities across price tiers,
    # different unit sizes -- not just random rows
    test_cases = [
        ("Dubai Marina", 700), ("Dubai Marina", 1400),
        ("Bur Dubai", 600), ("Emirates Hills", 3000),
        ("Deira", 550), ("Palm Jumeirah", 2000),
        ("Jumeirah Village Circle", 800), ("Business Bay", 1000),
    ][:n]

    for community, size in test_cases:
        recent = market_df[market_df["community_key"] == community.lower()].sort_values("year_month")
        if recent.empty:
            print(f"{community:30s} | NOT FOUND in dataset")
            continue
        actual_rent_per_sqft = recent.iloc[-1]["rental_price_per_sqft_annual_usd"]

        prop = PropertyInput(community=community, size_sqft=size, purchase_price_usd=None)
        pred_rent_per_sqft, n_comps, source = estimate_rent(
            prop, market_df, model, spec["numeric"], spec["categorical"]
        )
        error_pct = (pred_rent_per_sqft - actual_rent_per_sqft) / actual_rent_per_sqft * 100
        print(f"{community:25s} | {size:5.0f}sqft | actual: ${actual_rent_per_sqft:6.2f}/sqft | "
              f"predicted: ${pred_rent_per_sqft:6.2f}/sqft | error: {error_pct:+6.1f}% | n_comps={n_comps}")


def check_price(n=8):
    print("\n" + "=" * 70)
    print("PRICE: diverse areas/sizes/bedrooms, predicted vs actual (real test rows)")
    print("=" * 70)

    test = pd.read_csv("data/processed_price/test.csv")
    explainer = PriceExplainer(model_dir="models/price", data_dir="data/processed_price")

    # Pick diverse real rows: different areas across price tiers, different
    # bedroom counts, different years -- not just one random sample
    diverse_areas = ["Al Wasl", "Al Warsan First", "Dubai Marina", "Business Bay",
                      "Jumeirah Village Circle", "Downtown Dubai", "Palm Deira", "Al Barsha"]
    picked_rows = []
    for area in diverse_areas[:n]:
        subset = test[test["area_name"] == area]
        if not subset.empty:
            picked_rows.append(subset.sample(1, random_state=1).iloc[0])

    if not picked_rows:
        print("None of the target areas found in test set -- falling back to random sample")
        picked_rows = [row for _, row in test.sample(n, random_state=1).iterrows()]

    for row in picked_rows:
        property_row = {
            "area_name": row["area_name"], "property_sub_type": row["property_sub_type"],
            "area_sqm": row["area_sqm"], "bedrooms": row["bedrooms"],
            "has_parking": row["has_parking"], "near_landmark": row["near_landmark"],
            "near_mall": row["near_mall"], "near_metro": row["near_metro"],
            "is_offplan": row["is_offplan"], "year": row["year"],
            "month_sin": row["month_sin"], "month_cos": row["month_cos"], "t_index": row["t_index"],
        }
        result = explainer.predict_and_explain(property_row, top_n=2)
        actual = row["price"]
        pred = result["calibrated_predicted_price"]
        error_pct = (pred - actual) / actual * 100
        print(f"{row['area_name']:25s} | {row['area_sqm']:6.0f}sqm | {row['bedrooms']:.0f}BR | "
              f"actual: AED {actual:>10,.0f} | predicted: AED {pred:>10,.0f} | "
              f"error: {error_pct:+6.1f}% | confidence: {result['confidence']}")


def check_investment_scores(n=8):
    print("\n" + "=" * 70)
    print("INVESTMENT SCORE: diverse communities (no ground truth -- sanity check spread)")
    print("=" * 70)

    market_df = pd.read_csv("data/processed/full_processed.csv", parse_dates=["year_month"])
    with open("data/processed/feature_columns.json") as f:
        spec = json.load(f)
    model = joblib.load("models/random_forest_model.pkl")

    communities = ["Dubai Marina", "Bur Dubai", "Emirates Hills", "Deira",
                    "Palm Jumeirah", "Jumeirah Village Circle", "Business Bay", "The Valley"][:n]

    for community in communities:
        recent = market_df[market_df["community_key"] == community.lower()].sort_values("year_month")
        if recent.empty:
            continue
        price_per_sqft = recent.iloc[-1]["secondary_price_per_sqft_usd"]
        size = 1000
        prop = PropertyInput(community=community, size_sqft=size, purchase_price_usd=price_per_sqft * size)
        result = compute_yield_and_score(prop, market_df, model, spec["numeric"], spec["categorical"])
        print(f"{community:25s} | gross yield: {result.gross_yield_pct:5.2f}% | "
              f"score: {result.investment_score:5.1f}/100 | "
              f"demand: {result.score_breakdown['demand_score']:5.1f} | "
              f"momentum: {result.score_breakdown['price_momentum_score']:5.1f}")


if __name__ == "__main__":
    check_rent()
    check_price()
    check_investment_scores()
    print("\n" + "=" * 70)
    print("Done. Review each section for: (1) errors within a reasonable range "
          "(rent ~single-digit %, price ~15-20% MAPE-consistent), (2) score "
          "spread that actually differs across communities, not flat.")