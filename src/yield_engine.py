"""
yield_engine.py
----------------
Rental yield & investment-scoring engine.

Takes a property (price, size, community) plus the market comparables
(from the processed rent panel) and computes:
    - Predicted/comparable annual rent (via matched community + model)
    - Gross rental yield
    - Net rental yield (after standard operating costs)
    - Simple ROI over a holding period
    - A composite 0-100 investment score blending yield, market demand,
      price trend, and data confidence

This module does NOT retrain anything -- it consumes:
    1. A trained rent-prediction model (from train.py / models/)
    2. The processed market panel (data/processed/full_processed.csv)
       as the source of comparables

Design choice: yield is computed primarily from REAL comparables (the
matched community's actual recent rent), with the ML prediction used as
a cross-check / fallback when comparable data is thin. This mirrors the
"hybrid ML + benchmark" approach from the price project, and for the
same reason: pure model output on a single data point is risky, but
market comparables are ground truth.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---- Default assumptions (override per-property as needed) ----
DEFAULT_SERVICE_CHARGE_PCT = 0.12   # % of annual rent lost to service/maintenance charges
DEFAULT_VACANCY_PCT = 0.05          # % of year assumed vacant between tenants
DEFAULT_MANAGEMENT_FEE_PCT = 0.05   # % of annual rent for property management
DEFAULT_TRANSACTION_COST_PCT = 0.06 # DLD fee + agent fee etc., for ROI-on-cost calcs


@dataclass
class PropertyInput:
    community: str
    size_sqft: float
    purchase_price_usd: Optional[float] = None  # None when only estimating rent, no price known
    zone: Optional[str] = None
    is_freehold: Optional[bool] = None
    holding_period_years: float = 5.0
    service_charge_pct: float = DEFAULT_SERVICE_CHARGE_PCT
    vacancy_pct: float = DEFAULT_VACANCY_PCT
    management_fee_pct: float = DEFAULT_MANAGEMENT_FEE_PCT
    transaction_cost_pct: float = DEFAULT_TRANSACTION_COST_PCT


@dataclass
class YieldResult:
    community_key: str
    matched: bool
    comparable_rent_per_sqft: float
    predicted_annual_rent_usd: float
    gross_yield_pct: float
    net_yield_pct: float
    roi_5yr_pct: float
    investment_score: float
    score_breakdown: dict = field(default_factory=dict)
    n_comparables: int = 0
    confidence: str = "low"
    notes: list = field(default_factory=list)


def _normalize(s: str) -> str:
    return str(s).strip().lower()


def get_community_comps(market_df: pd.DataFrame, community: str, lookback_months: int = 6) -> pd.DataFrame:
    """Pull the most recent N months of real market rows for a community.
    This is the 'real rental market comparables' the user's spec refers to."""
    key = _normalize(community)
    sub = market_df[market_df["community_key"] == key].copy()
    if sub.empty:
        return sub
    sub = sub.sort_values("year_month")
    return sub.tail(lookback_months)


def estimate_rent(prop: PropertyInput, market_df: pd.DataFrame, model=None,
                   model_numeric_cols=None, model_categorical_cols=None) -> tuple:
    """Returns (rent_per_sqft, n_comparables, source_note).

    Priority:
      1. Real comparables from the same community, recent months -> weighted
         average (more recent months weighted higher).
      2. If comparables are thin (< 3 rows), blend with the ML model's
         prediction (60% comps / 40% model) if a model is supplied.
      3. If no comparables at all, fall back fully to the ML model.
    """
    comps = get_community_comps(market_df, prop.community)
    n = len(comps)

    if n >= 3:
        weights = np.linspace(1, 2, n)  # more recent months weigh more
        comp_rent = np.average(comps["rental_price_per_sqft_annual_usd"], weights=weights)
        if n >= 6 or model is None:
            return comp_rent, n, f"comparables (n={n})"
        # thin-ish but usable: blend with model
        model_rent = _predict_with_model(prop, comps.iloc[-1], model, model_numeric_cols, model_categorical_cols, market_df)
        blended = 0.6 * comp_rent + 0.4 * model_rent
        return blended, n, f"blended: 60% comps (n={n}) + 40% model"

    if n > 0 and model is not None:
        comp_rent = comps["rental_price_per_sqft_annual_usd"].mean()
        model_rent = _predict_with_model(prop, comps.iloc[-1], model, model_numeric_cols, model_categorical_cols, market_df)
        blended = 0.5 * comp_rent + 0.5 * model_rent
        return blended, n, f"blended: 50% comps (n={n}, thin) + 50% model"

    if model is not None:
        # No comps at all -- need a representative row for the model's other
        # features (macro rates etc). Use the most recent row in the whole
        # market as a fallback context, but flag it clearly.
        fallback_row = market_df.sort_values("year_month").iloc[-1]
        model_rent = _predict_with_model(prop, fallback_row, model, model_numeric_cols, model_categorical_cols, market_df)
        return model_rent, 0, "model only, NO community comparables found -- low confidence"

    raise ValueError(f"No comparables found for community '{prop.community}' and no model provided.")


def _predict_with_model(prop, context_row, model, numeric_cols, categorical_cols, market_df=None):
    row = context_row.copy()
    row["community_key"] = _normalize(prop.community)
    if prop.zone:
        row["zone"] = prop.zone
    if prop.is_freehold is not None:
        row["is_freehold"] = prop.is_freehold

    if prop.purchase_price_usd and prop.size_sqft and prop.size_sqft > 0:
        known_price_per_sqft = prop.purchase_price_usd / prop.size_sqft
        row["secondary_price_per_sqft_usd"] = known_price_per_sqft
        row["offplan_price_per_sqft_usd"] = known_price_per_sqft
        row["offplan_secondary_price_gap"] = 0.0
        # 1 sqft = 0.092903 m2 -- this column was still carrying the
        # unrelated fallback row's price-per-m2, contradicting the
        # price-per-sqft we just set above. Same bug, different unit.
        if "secondary_price_per_m2_usd" in numeric_cols:
            row["secondary_price_per_m2_usd"] = known_price_per_sqft / 0.092903

        # Also fix the rent-history features (lag/rolling/ratio). Left
        # untouched, these still carry an UNRELATED community's actual
        # rent level, which contradicts the price we just set and drags
        # the prediction toward whatever community the context row came
        # from (this was the residual cause of the 17% yield result --
        # price said "budget", rent history said "luxury").
        # Instead, derive a synthetic, self-consistent rent history from
        # the known price and the market-wide average yield, so every
        # feature the model sees agrees with each other.
        lag_cols = [c for c in numeric_cols if c.startswith("rent_lag") or c.startswith("rent_roll")]
        if lag_cols and market_df is not None:
            avg_yield_fraction = (
                market_df["rental_price_per_sqft_annual_usd"] / market_df["secondary_price_per_sqft_usd"]
            ).mean()
            synthetic_rent = known_price_per_sqft * avg_yield_fraction
            for c in lag_cols:
                row[c] = synthetic_rent
            if "price_to_rent_ratio_lag1" in numeric_cols:
                row["price_to_rent_ratio_lag1"] = known_price_per_sqft / synthetic_rent

    X = pd.DataFrame([row[numeric_cols + categorical_cols]])
    return float(model.predict(X)[0])


def compute_yield_and_score(prop: PropertyInput, market_df: pd.DataFrame,
                             model=None, model_numeric_cols=None,
                             model_categorical_cols=None) -> YieldResult:
    if not prop.purchase_price_usd or prop.purchase_price_usd <= 0:
        raise ValueError("purchase_price_usd is required (and must be > 0) to compute yield/ROI/score.")

    notes = []
    rent_per_sqft, n_comps, source_note = estimate_rent(
        prop, market_df, model, model_numeric_cols, model_categorical_cols
    )
    notes.append(f"Rent source: {source_note}")

    annual_rent = rent_per_sqft * prop.size_sqft

    # --- Gross yield ---
    gross_yield = (annual_rent / prop.purchase_price_usd) * 100

    # --- Net yield: subtract service charges, vacancy loss, management fee ---
    effective_rent = annual_rent * (1 - prop.vacancy_pct)
    costs = annual_rent * (prop.service_charge_pct + prop.management_fee_pct)
    net_annual_income = effective_rent - costs
    net_yield = (net_annual_income / prop.purchase_price_usd) * 100

    # --- Simple ROI over holding period (rental income only, no appreciation) ---
    total_cost_basis = prop.purchase_price_usd * (1 + prop.transaction_cost_pct)
    cumulative_net_income = net_annual_income * prop.holding_period_years
    roi_pct = (cumulative_net_income / total_cost_basis) * 100

    # --- Composite investment score (0-100) ---
    score, breakdown = _investment_score(
        gross_yield=gross_yield,
        net_yield=net_yield,
        n_comps=n_comps,
        market_df=market_df,
        community=prop.community,
    )

    confidence = "high" if n_comps >= 6 else ("medium" if n_comps >= 3 else "low")
    if confidence == "low":
        notes.append("Low confidence: few or no direct market comparables for this community/period.")

    return YieldResult(
        community_key=_normalize(prop.community),
        matched=n_comps > 0,
        comparable_rent_per_sqft=round(rent_per_sqft, 2),
        predicted_annual_rent_usd=round(annual_rent, 2),
        gross_yield_pct=round(gross_yield, 2),
        net_yield_pct=round(net_yield, 2),
        roi_5yr_pct=round(roi_pct, 2),
        investment_score=round(score, 1),
        score_breakdown=breakdown,
        n_comparables=n_comps,
        confidence=confidence,
        notes=notes,
    )


def _investment_score(gross_yield, net_yield, n_comps, market_df, community) -> tuple:
    """Composite 0-100 score blending:
      - Net yield (20%)      : profitability signal -- LOWER weight than
                                you'd expect, because in this dataset yield
                                is nearly flat across communities (std <0.1pp
                                on a ~6.5% mean, confirmed via correlation
                                check: price vs rent corr = 0.9999). It still
                                counts, but it can't do much discriminating
                                work on its own here.
      - Demand trend (35%)   : rising rental listing activity = demand.
                                This DOES vary meaningfully across communities
                                and time, so it carries more weight.
      - Price momentum (30%) : is the community's sale price trending up.
                                Also a genuinely differentiating signal.
      - Data confidence (15%): penalize thin-comparable estimates
    Each component is scored 0-100 then weighted.

    NOTE: if you later swap in a dataset where yield genuinely varies by
    community (e.g. real DLD + real rental listings, rather than this
    formula-generated panel), bump the yield weight back up -- these
    weights are tuned to what this specific dataset can actually tell you,
    not a universal truth about Dubai real estate.
    """
    key = _normalize(community)
    comm_hist = market_df[market_df["community_key"] == key].sort_values("year_month")

    # 1. Net yield score: scale against the REAL observed market band for
    #    net yield (derived from market_df), not a guessed 2-9% range.
    #    This makes the score actually spread across communities as widely
    #    as the real data allows, instead of compressing everything into a
    #    narrow slice of 0-100 because the assumed band was too wide.
    market_yields = (
        market_df["rental_price_per_sqft_annual_usd"] / market_df["secondary_price_per_sqft_usd"] * 100
    )
    band_low, band_high = market_yields.quantile(0.05), market_yields.quantile(0.95)
    if band_high <= band_low:
        yield_score = 50.0  # degenerate case, fall back to neutral
    else:
        # Use gross_yield here, not net_yield -- the band above is built
        # from market-wide GROSS yield (rent/price), so comparing net
        # yield (which is always lower, after costs) against it would
        # clip to 0 every time. Net yield still gets reported to the user
        # separately; it's just not what drives this component's score.
        yield_score = np.clip((gross_yield - band_low) / (band_high - band_low) * 100, 0, 100)

    # 2. Demand trend: compare rental listings in the most recent 6 months
    #    vs the prior 6 months
    demand_score = 50.0  # neutral default
    if len(comm_hist) >= 12:
        recent = comm_hist["n_listings_rental"].tail(6).mean()
        prior = comm_hist["n_listings_rental"].iloc[-12:-6].mean()
        if prior > 0:
            growth = (recent - prior) / prior
            demand_score = np.clip(50 + growth * 100, 0, 100)

    # 3. Price momentum: sale price trend over the last 12 months
    price_score = 50.0
    if len(comm_hist) >= 12:
        recent_price = comm_hist["secondary_price_per_sqft_usd"].tail(6).mean()
        prior_price = comm_hist["secondary_price_per_sqft_usd"].iloc[-12:-6].mean()
        if prior_price > 0:
            price_growth = (recent_price - prior_price) / prior_price
            price_score = np.clip(50 + price_growth * 150, 0, 100)

    # 4. Confidence score: more comparables = more trustworthy
    confidence_score = np.clip(n_comps / 6 * 100, 0, 100)

    weights = {"yield": 0.20, "demand": 0.35, "price_momentum": 0.30, "confidence": 0.15}
    composite = (
        yield_score * weights["yield"]
        + demand_score * weights["demand"]
        + price_score * weights["price_momentum"]
        + confidence_score * weights["confidence"]
    )

    breakdown = {
        "yield_score": round(float(yield_score), 1),
        "demand_score": round(float(demand_score), 1),
        "price_momentum_score": round(float(price_score), 1),
        "confidence_score": round(float(confidence_score), 1),
        "weights": weights,
    }
    return composite, breakdown


if __name__ == "__main__":
    # Quick smoke test with dummy data
    market_df = pd.DataFrame({
        "year_month": pd.date_range("2024-01-01", periods=18, freq="MS"),
        "community_key": ["palm jumeirah"] * 18,
        "rental_price_per_sqft_annual_usd": np.linspace(40, 46, 18),
        "secondary_price_per_sqft_usd": np.linspace(650, 700, 18),
        "n_listings_rental": np.linspace(60, 90, 18),
    })
    prop = PropertyInput(community="Palm Jumeirah", size_sqft=1200, purchase_price_usd=850000)
    result = compute_yield_and_score(prop, market_df)
    print(json.dumps(result.__dict__, indent=2, default=str))