"""
api/main.py
------------
FastAPI backend exposing:
    POST /predict-rent   -> raw rent-per-sqft prediction for a community
    POST /yield-score    -> full gross/net yield, ROI, investment score
    GET  /communities    -> list of known communities (for frontend dropdowns)
    GET  /health

Run with:
    uvicorn api.main:app --reload --port 8000
"""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# allow importing from src/ when running as `uvicorn api.main:app` from project root
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src", "price"))
from src.yield_engine import PropertyInput, compute_yield_and_score  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

app = FastAPI(title="PropLens Rent & Yield API", version="0.1.0")

# --- Load artifacts once at startup ---
_market_df = None
_model = None
_feature_spec = None
_price_explainer = None

PRICE_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed_price")
PRICE_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "price")


@app.on_event("startup")
def load_artifacts():
    global _market_df, _model, _feature_spec, _price_explainer

    market_path = os.path.join(DATA_DIR, "full_processed.csv")
    if not os.path.exists(market_path):
        raise RuntimeError(f"Market data not found at {market_path}. Run src/data_pipeline.py first.")
    _market_df = pd.read_csv(market_path, parse_dates=["year_month"])

    spec_path = os.path.join(DATA_DIR, "feature_columns.json")
    with open(spec_path) as f:
        _feature_spec = json.load(f)

    best_path = os.path.join(MODEL_DIR, "best_model.json")
    if os.path.exists(best_path):
        with open(best_path) as f:
            best = json.load(f)
        best_name = best["best_model"]
        if best_name == "catboost":
            from catboost import CatBoostRegressor
            m = CatBoostRegressor()
            m.load_model(os.path.join(MODEL_DIR, "catboost_model.cbm"))
            _model = m
        else:
            _model = joblib.load(os.path.join(MODEL_DIR, f"{best_name}_model.pkl"))
        print(f"Loaded model: {best_name}")
    else:
        print("WARNING: no trained model found -- yield engine will rely on comparables only.")

    # --- Price prediction model (separate module, loaded here so it's
    #     ready once and reused across requests, not rebuilt per call) ---
    try:
        from explain import PriceExplainer
        _price_explainer = PriceExplainer(model_dir=PRICE_MODEL_DIR, data_dir=PRICE_DATA_DIR)
        print("Loaded price prediction model + SHAP explainer")
    except Exception as e:
        print(f"WARNING: price model not loaded ({e}). /predict-price will be unavailable "
              f"until src/price/train.py and calibrate.py have been run.")


# --- Request/response schemas ---
class RentPredictRequest(BaseModel):
    community: str
    size_sqft: float = Field(gt=0)
    zone: str | None = None
    is_freehold: bool | None = None


class YieldRequest(BaseModel):
    community: str
    size_sqft: float = Field(gt=0)
    purchase_price_usd: float = Field(gt=0)
    zone: str | None = None
    is_freehold: bool | None = None
    holding_period_years: float = 5.0
    service_charge_pct: float = 0.12
    vacancy_pct: float = 0.05
    management_fee_pct: float = 0.05
    transaction_cost_pct: float = 0.06


class PricePredictRequest(BaseModel):
    area_name: str
    property_sub_type: str = "Flat"
    area_sqm: float = Field(gt=15, le=1500)  # matches training data sanity bounds
    bedrooms: float = 1
    has_parking: bool = True
    near_landmark: bool = False
    near_mall: bool = False
    near_metro: bool = False
    is_offplan: bool = False
    year: int = 2026
    month: int = Field(default=6, ge=1, le=12)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "market_rows": len(_market_df) if _market_df is not None else 0}


@app.get("/communities")
def list_communities():
    if _market_df is None:
        raise HTTPException(500, "Market data not loaded")
    coms = (
        _market_df[["community", "community_key", "zone"]]
        .drop_duplicates()
        .sort_values("community")
        .to_dict(orient="records")
    )
    return {"communities": coms, "count": len(coms)}


@app.post("/predict-rent")
def predict_rent(req: RentPredictRequest):
    from src.yield_engine import estimate_rent

    prop = PropertyInput(
        community=req.community, size_sqft=req.size_sqft,
        purchase_price_usd=None,  # rent estimate should not depend on a made-up price
        zone=req.zone, is_freehold=req.is_freehold,
    )
    rent_per_sqft, n_comps, source = estimate_rent(
        prop, _market_df, _model,
        _feature_spec["numeric"], _feature_spec["categorical"],
    )
    return {
        "community": req.community,
        "rent_per_sqft_annual_usd": round(rent_per_sqft, 2),
        "estimated_annual_rent_usd": round(rent_per_sqft * req.size_sqft, 2),
        "n_comparables": n_comps,
        "source": source,
    }


@app.post("/yield-score")
def yield_score(req: YieldRequest):
    prop = PropertyInput(**req.dict())
    try:
        result = compute_yield_and_score(
            prop, _market_df, _model,
            _feature_spec["numeric"], _feature_spec["categorical"],
        )
    except ValueError as e:
        raise HTTPException(404, str(e))
    return result.__dict__


@app.get("/price-areas")
def list_price_areas():
    if _price_explainer is None:
        raise HTTPException(503, "Price model not loaded.")
    areas = sorted(_price_explainer.area_stats.keys())
    return {"areas": areas, "count": len(areas)}


@app.post("/predict-price")
def predict_price(req: PricePredictRequest):
    if _price_explainer is None:
        raise HTTPException(503, "Price model not loaded. Run src/price/train.py "
                                  "and src/price/calibrate.py first.")

    base_year = _price_explainer.base_year
    t_index = (req.year - base_year) * 12 + req.month
    month_sin = float(np.sin(2 * np.pi * req.month / 12))
    month_cos = float(np.cos(2 * np.pi * req.month / 12))

    property_row = {
        "area_name": req.area_name,
        "property_sub_type": req.property_sub_type,
        "area_sqm": req.area_sqm,
        "bedrooms": req.bedrooms,
        "has_parking": int(req.has_parking),
        "near_landmark": int(req.near_landmark),
        "near_mall": int(req.near_mall),
        "near_metro": int(req.near_metro),
        "is_offplan": int(req.is_offplan),
        "year": req.year,
        "month_sin": month_sin,
        "month_cos": month_cos,
        "t_index": t_index,
    }

    try:
        result = _price_explainer.predict_and_explain(property_row)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")

    return result