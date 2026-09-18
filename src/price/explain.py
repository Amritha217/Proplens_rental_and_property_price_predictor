"""
src/price/explain.py
----------------------
Per-prediction SHAP explainability: for any single predicted price, shows
which features pushed it up or down and by how much. This is what makes
the model interpretable rather than a black box (resume bullet #3).

Design choice: SHAP values are computed PER PREDICTION at inference time
(fast -- a handful of milliseconds for one row), not batch-computed over
the whole 250K-row test set (which would be slow and isn't what the API
needs). A background sample is used once at startup for the explainer's
baseline, not recomputed per request.

Usage (standalone test):
    python src/price/explain.py --data_dir data/processed_price --model_dir models/price
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap


class PriceExplainer:
    def __init__(self, model_dir="models/price", data_dir="data/processed_price", background_size=200):
        with open(os.path.join(model_dir, "best_model.json")) as f:
            self.best = json.load(f)
        self.model_pipeline = joblib.load(
            os.path.join(model_dir, f"{self.best['best_model']}_full_model.pkl")
        )
        self.feature_cols = list(self.model_pipeline.named_steps["pre"].feature_names_in_)
        self.preprocessor = self.model_pipeline.named_steps["pre"]
        self.core_model = self.model_pipeline.named_steps["model"]

        with open(os.path.join(model_dir, "area_encoding_map.json")) as f:
            self.area_enc = json.load(f)
        with open(os.path.join(model_dir, "trend_calibration.json")) as f:
            self.calib = json.load(f)
        with open(os.path.join(data_dir, "feature_columns.json")) as f:
            self.base_year = json.load(f)["base_year"]

        # Confidence check inputs: how many training rows exist per area,
        # and what size range that area's training data actually covers.
        # Used to flag cold-start areas (zero training history, like
        # "Palm Jabal Ali" -- a newly-transacting area) and extrapolation
        # cases (a requested size well outside what that area's training
        # data ever covered, like an oversized unit in a small-sample area).
        train_raw = pd.read_csv(os.path.join(data_dir, "train.csv"))
        self.area_stats = train_raw.groupby("area_name")["area_sqm"].agg(
            n="count", sqm_min="min", sqm_max="max"
        ).to_dict(orient="index")

        # Background sample for the SHAP explainer's baseline expectation.
        # Drawn once from training data, reused for every explanation --
        # NOT recomputed per request (that would be slow and pointless,
        # the baseline shouldn't change prediction to prediction).
        train = pd.read_csv(os.path.join(data_dir, "train.csv")).dropna(subset=self.feature_cols_raw_needed())
        train["area_target_enc"] = train["area_name"].map(self.area_enc["map"]).fillna(self.area_enc["global_mean"])
        background = train[self.feature_cols].sample(min(background_size, len(train)), random_state=42)
        background_transformed = self.preprocessor.transform(background)

        # TreeExplainer is fast and exact for tree models (LightGBM/RF/XGBoost/CatBoost)
        self.explainer = shap.TreeExplainer(self.core_model, background_transformed)
        self.transformed_feature_names = self._get_transformed_feature_names()

    def feature_cols_raw_needed(self):
        # numeric + categorical columns as they exist pre-encoding
        return [c for c in self.feature_cols if c != "area_target_enc"] + ["area_name"]

    def _get_transformed_feature_names(self):
        try:
            return list(self.preprocessor.get_feature_names_out())
        except Exception:
            return [f"f{i}" for i in range(self.preprocessor.transform(
                pd.DataFrame([{c: 0 for c in self.feature_cols}])
            ).shape[1])]

    def predict_batch(self, df: pd.DataFrame):
        """Fast bulk prediction + confidence, WITHOUT per-row SHAP (SHAP is
        for single-prediction explainability at inference time -- running
        it over thousands of rows here would be slow and isn't needed for
        a bulk accuracy check)."""
        df = df.copy()
        df["area_target_enc"] = df["area_name"].map(self.area_enc["map"]).fillna(self.area_enc["global_mean"])
        X = df[self.feature_cols]

        raw_preds = self.model_pipeline.predict(X)
        months_ahead = np.clip(df["t_index"].values - self.calib["max_train_t_index"], a_min=0, a_max=None)
        factor = np.exp(self.calib["monthly_log_growth"] * months_ahead)
        calibrated_preds = raw_preds * factor

        def get_confidence(row):
            stats = self.area_stats.get(row["area_name"])
            if stats is None or stats["n"] == 0:
                return "low"
            n = stats["n"]
            buffer_low, buffer_high = stats["sqm_min"] * 0.7, stats["sqm_max"] * 1.3
            if n < 20 or not (buffer_low <= row["area_sqm"] <= buffer_high):
                return "low"
            return "medium" if n < 100 else "high"

        confidence = df.apply(get_confidence, axis=1)
        return raw_preds, calibrated_preds, confidence

    def predict_and_explain(self, property_row: dict, top_n=5):
        """property_row: dict with keys matching the raw feature columns
        (area_sqm, bedrooms, has_parking, near_landmark, near_mall,
        near_metro, is_offplan, year, month_sin, month_cos, t_index,
        area_name, property_sub_type)."""
        row = property_row.copy()
        row["area_target_enc"] = self.area_enc["map"].get(
            row.get("area_name"), self.area_enc["global_mean"]
        )
        X = pd.DataFrame([row])[self.feature_cols]

        raw_pred = float(self.model_pipeline.predict(X)[0])

        # calibration for out-of-training-range years, same as calibrate.py
        months_ahead = max(0, row.get("t_index", 0) - self.calib["max_train_t_index"])
        calibration_factor = float(np.exp(self.calib["monthly_log_growth"] * months_ahead))
        calibrated_pred = raw_pred * calibration_factor

        X_transformed = self.preprocessor.transform(X)
        shap_values = self.explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):  # some model/version combos return a list
            shap_values = shap_values[0]
        shap_row = shap_values[0]

        contributions = sorted(
            zip(self.transformed_feature_names, shap_row),
            key=lambda x: abs(x[1]), reverse=True
        )[:top_n]

        explanation = [
            {"feature": name, "impact_aed": round(float(val), 0),
             "direction": "increases price" if val > 0 else "decreases price"}
            for name, val in contributions
        ]

        # --- Confidence assessment ---
        area_name = row.get("area_name")
        requested_sqm = row.get("area_sqm")
        stats = self.area_stats.get(area_name)
        notes = []

        if stats is None or stats["n"] == 0:
            confidence = "low"
            notes.append(f"'{area_name}' has NO training history -- prediction relies "
                          f"entirely on the global average and other features, not this "
                          f"area's actual price level. Likely a newly-transacting area.")
        else:
            n = stats["n"]
            sqm_min, sqm_max = stats["sqm_min"], stats["sqm_max"]
            # allow a reasonable buffer beyond the observed range before flagging
            buffer_low, buffer_high = sqm_min * 0.7, sqm_max * 1.3
            out_of_range = not (buffer_low <= requested_sqm <= buffer_high)

            if n < 20:
                confidence = "low"
                notes.append(f"Only {n} training examples for '{area_name}' -- "
                              f"prediction may be unreliable.")
            elif out_of_range:
                confidence = "low"
                notes.append(f"Requested size ({requested_sqm} sqm) is well outside "
                              f"'{area_name}''s training range ({sqm_min:.0f}-{sqm_max:.0f} sqm) "
                              f"-- the model is extrapolating, which tree models handle poorly.")
            elif n < 100:
                confidence = "medium"
            else:
                confidence = "high"

        return {
            "raw_predicted_price": round(raw_pred, 0),
            "calibrated_predicted_price": round(calibrated_pred, 0),
            "calibration_factor": round(calibration_factor, 4),
            "base_value_aed": round(float(self.explainer.expected_value), 0),
            "top_price_drivers": explanation,
            "confidence": confidence,
            "notes": notes,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed_price")
    parser.add_argument("--model_dir", default="models/price")
    args = parser.parse_args()

    explainer = PriceExplainer(args.model_dir, args.data_dir)

    sample_property = {
        "area_name": "Al Wasl", "property_sub_type": "Flat",
        "area_sqm": 90, "bedrooms": 2, "has_parking": 1,
        "near_landmark": 1, "near_mall": 1, "near_metro": 0,
        "is_offplan": 0, "year": 2026, "month_sin": 0.5, "month_cos": 0.87,
        "t_index": explainer.calib["max_train_t_index"] + 6,
    }
    result = explainer.predict_and_explain(sample_property)
    print(json.dumps(result, indent=2))