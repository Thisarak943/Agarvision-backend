from __future__ import annotations

import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple

import joblib
import numpy as np
import pandas as pd


# ---------- Current selling prices (given by you) ----------
CURRENT_PRICES_LKR = {
    "Silani Ravana": 52628,
    "Silani Savera": 44061,
    "Silani Cobra": 14042,
    "Silani Junglefowl": 26314,
    "Silani Butterfly": 26314,
    "Silani Peacock": 26314,
}


# ---------- Demand level rules ----------
def demand_level_from_index(demand_index: float) -> str:
    if demand_index >= 70:
        return "High"
    if demand_index >= 40:
        return "Medium"
    return "Low"


# ---------- Price recommendation rules (simple + correct dependency on demand) ----------
def recommend_price_range(base_price: float, demand_index: float, festival_season: bool) -> Tuple[int, int]:
    """
    Rule-based:
      High demand -> higher min & max
      Medium -> slight increase
      Low -> decrease
    """
    if demand_index >= 70:        # High
        min_mult, max_mult = 1.02, 1.08
    elif demand_index >= 40:      # Medium
        min_mult, max_mult = 0.98, 1.04
    else:                         # Low
        min_mult, max_mult = 0.90, 0.98

    # Festival season small uplift
    if festival_season:
        min_mult += 0.01
        max_mult += 0.01

    min_price = int(round(base_price * min_mult / 10) * 10)
    max_price = int(round(base_price * max_mult / 10) * 10)

    # safety
    if max_price < min_price:
        max_price = min_price + 10

    return min_price, max_price


# ---------- Helpers to derive date features ----------
def derive_year_month_quarter(prediction_period: str) -> Tuple[int, int, int]:
    now = datetime.now()

    p = prediction_period.strip().lower()
    if "week" in p:
        future = now + timedelta(days=7)
    elif "quarter" in p:
        future = now + timedelta(days=90)
    else:  # default "month"
        future = now + timedelta(days=30)

    year = future.year
    month = future.month
    quarter = (month - 1) // 3 + 1
    return year, month, quarter


# ---------- Main predictor class ----------
class DemandPredictor:
    def __init__(self):
        base_dir = os.path.dirname(__file__)
        model_dir = os.path.join(base_dir, "model")

        self.model_path = os.path.join(model_dir, "demand_model.pkl")
        self.features_path = os.path.join(model_dir, "demand_features.pkl")
        self.encoders_path = os.path.join(model_dir, "label_encoders.pkl")

        self.model = None
        self.features: List[str] = []
        self.label_encoders: Dict[str, Any] = {}

        self._load()

    def _load(self):
        # model
        self.model = joblib.load(self.model_path)

        # features (list)
        self.features = joblib.load(self.features_path)

        # encoders (dict of LabelEncoder)
        with open(self.encoders_path, "rb") as f:
            self.label_encoders = pickle.load(f)

    def _encode_value(self, col: str, value: str) -> int:
        """
        Encode using saved LabelEncoder.
        If unseen value arrives, raise a clear error (frontend can show a message).
        """
        le = self.label_encoders.get(col)
        if le is None:
            # If a column was not encoded in training, return as-is
            return value

        if value not in le.classes_:
            allowed = list(map(str, le.classes_[:50]))
            raise ValueError(
                f"Invalid value for '{col}': '{value}'. Allowed examples: {allowed} ..."
            )

        return int(le.transform([value])[0])

    def _build_feature_row(self, payload: Dict[str, Any]) -> pd.DataFrame:
        # Frontend fields (exact)
        oil_type = payload["oil_type"]
        oil_grade = payload["oil_grade"]
        region = payload["market_region"]
        country = payload["market_country"]
        prediction_period = payload["prediction_period"]
        festival_season = bool(payload["festival_season"])

        # Derive date features
        year, month, quarter = derive_year_month_quarter(prediction_period)

        # Dynamic features based on market context
        # Buyer type varies by region (uses only valid training categories)
        buyer_type_by_region = {
            "Middle East": "Retail Boutique",
            "East Asia": "Distributor",
            "Southeast Asia": "Wholesaler",
            "Europe": "Perfume Manufacturer",
            "South Asia": "Distributor"
        }
        buyer_type = buyer_type_by_region.get(region, "Retail Boutique")
        
        # Substitute availability varies by region
        substitute_by_region = {
            "Middle East": "Low",      # High demand, fewer substitutes
            "East Asia": "Medium",
            "Southeast Asia": "High",  # More competition
            "Europe": "Medium",
            "South Asia": "High"
        }
        substitute_availability = substitute_by_region.get(region, "Medium")
        
        # Exchange rate varies by quarter (more realistic)
        exchange_rate = 295.0 + (quarter * 2.5)  # Slight variation by quarter

        # Grade -> aging & quality with variation by oil type
        grade_key = oil_grade.strip().lower()
        base_aging = {"premium": 12, "standard": 6}.get(grade_key, 3)
        base_quality = {"premium": 90, "standard": 75}.get(grade_key, 60)
        
        # Premium oils (Ravana, Savera) get slight boost
        oil_boost = 2 if oil_type in ["Silani Ravana", "Silani Savera"] else 0
        
        # Add variation based on time (seasonal effect)
        seasonal_quality_boost = 5 if quarter in [4, 1] else 0  # Q4 and Q1 boost
        
        aging_months = base_aging + oil_boost
        quality_score = min(100, base_quality + (oil_boost * 3) + seasonal_quality_boost)

        row = {
            "Year": year,
            "Month": month,
            "Quarter": quarter,
            "Festival_Season": int(festival_season),

            "Market_Region": region,
            "Market_Country": country,
            "Buyer_Type": buyer_type,
            "Oil_Type": oil_type,
            "Oil_Grade": oil_grade,

            "Aging_Months": aging_months,
            "Quality_Score": quality_score,
            "Substitute_Availability": substitute_availability,
            "Exchange_Rate_LKR_USD": exchange_rate,
        }

        # Ensure correct order (exact training feature order)
        df = pd.DataFrame([[row.get(f) for f in self.features]], columns=self.features)

        # Encode categorical columns exactly like training
        for col in df.select_dtypes(include=["object"]).columns.tolist():
            df[col] = self._encode_value(col, df.loc[0, col])

        return df

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        X = self._build_feature_row(payload)

        demand_index = float(self.model.predict(X)[0])
        demand_index = max(0.0, min(100.0, demand_index))  # clamp 0–100
        demand_index_rounded = int(round(demand_index))

        level = demand_level_from_index(demand_index)

        # Price recommendation using current selling price
        base_price = CURRENT_PRICES_LKR[payload["oil_type"]]
        min_p, max_p = recommend_price_range(
            base_price=base_price,
            demand_index=demand_index,
            festival_season=bool(payload["festival_season"]),
        )

        why = []
        if level == "High":
            why.append("Demand is expected to increase in this market.")
        elif level == "Medium":
            why.append("Demand is expected to be stable in this market.")
        else:
            why.append("Demand is expected to decrease in this market.")

        if bool(payload["festival_season"]):
            why.append("Festival season increases buying activity.")

        why.append("Predicted demand index indicates market movement.")

        return {
            "selected_details": {
                "oil_type": payload["oil_type"],
                "oil_grade": payload["oil_grade"],
                "market_country": payload["market_country"],
                "prediction_period": payload["prediction_period"],
            },
            "demand_index": demand_index_rounded,
            "demand_level": level,
            "recommended_price_range": {
                "min_price_lkr": min_p,
                "max_price_lkr": max_p,
            },
            "why": why,
        }