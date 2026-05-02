from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import load

from .constants import (
    COUNTRY_REASON_NOTES,
    COUNTRY_TO_REGION,
    CURRENT_OIL_PRICES_LKR,
    DEMAND_CATEGORY_THRESHOLDS,
    KEY_EXPORT_MARKET_NOTES,
    MODEL_INPUT_COLUMNS,
    MONTH_NUMBER_TO_NAME,
    OIL_TYPE_ALIASES,
    PRICE_RECOMMENDATION_RULES,
)


MODEL_PATH = Path(__file__).resolve().parent / "model" / "agarwood_demand_model.joblib"


def demand_category_from_index(demand_index: float) -> str:
    if demand_index <= DEMAND_CATEGORY_THRESHOLDS["low_max"]:
        return "Low"
    if demand_index <= DEMAND_CATEGORY_THRESHOLDS["medium_max"]:
        return "Medium"
    return "High"


def resolve_prediction_period(
    prediction_month: int,
    prediction_week: int,
    current_date: date | None = None,
) -> dict[str, Any]:
    today = current_date or date.today()
    prediction_year = today.year if prediction_month >= today.month else today.year + 1
    horizon_months = prediction_month - today.month
    if horizon_months < 0:
        horizon_months += 12

    if horizon_months == 0:
        horizon_label = "Current month"
    elif horizon_months == 1:
        horizon_label = "1 month ahead"
    else:
        horizon_label = f"{horizon_months} months ahead"

    return {
        "year": prediction_year,
        "month": prediction_month,
        "month_name": MONTH_NUMBER_TO_NAME[prediction_month],
        "week": prediction_week,
        "forecast_horizon_months": horizon_months,
        "forecast_horizon": horizon_label,
        "forecast_scope": "Next upcoming 12 months",
    }


def get_recommended_price_range(
    oil_type: str,
    demand_index: float,
    demand_category: str,
) -> dict[str, int]:
    base_price = CURRENT_OIL_PRICES_LKR[oil_type]
    low_max = DEMAND_CATEGORY_THRESHOLDS["low_max"]
    medium_max = DEMAND_CATEGORY_THRESHOLDS["medium_max"]

    if demand_category == "Low":
        rules = PRICE_RECOMMENDATION_RULES["Low"]
        low_strength = min(max((low_max - demand_index) / low_max, 0), 1)
        min_discount = rules["min_discount_percent"] + (low_strength * 3.0)
        max_discount = rules["max_discount_percent"]
        min_price = base_price * (1 - min_discount / 100)
        max_price = base_price * (1 - max_discount / 100)
    elif demand_category == "Medium":
        rules = PRICE_RECOMMENDATION_RULES["Medium"]
        min_price = base_price * (1 + rules["min_change_percent"] / 100)
        max_price = base_price * (1 + rules["max_change_percent"] / 100)
    else:
        rules = PRICE_RECOMMENDATION_RULES["High"]
        high_strength = min(max((demand_index - medium_max) / (88.0 - medium_max), 0), 1)
        max_premium = (
            rules["base_max_premium_percent"]
            + high_strength * rules["extra_max_premium_percent"]
        )
        min_premium = high_strength * rules["max_min_premium_percent"]
        min_price = base_price * (1 + min_premium / 100)
        max_price = base_price * (1 + max_premium / 100)

    return {
        "current_selling_price_lkr": base_price,
        "min_price_lkr": round(min_price),
        "max_price_lkr": round(max_price),
    }


def generate_prediction_reasons(
    input_data: dict[str, Any],
    demand_category: str,
) -> list[str]:
    oil_grade = str(input_data["Oil_Grade"])
    market_region = str(input_data["Market_Region"])
    market_country = str(input_data["Market_Country"])
    prediction_month = int(input_data["Prediction_month"])
    prediction_week = int(input_data["Prediction_week"])
    festival_season = int(input_data["Festival_Season"])

    market_reason = COUNTRY_REASON_NOTES.get(
        market_country,
        KEY_EXPORT_MARKET_NOTES.get(
            market_region,
            f"{market_country} market demand influenced this prediction.",
        ),
    )

    if demand_category == "High":
        reasons = []
        if oil_grade == "Premium":
            reasons.append("Premium grade oil is a major driver of the high demand prediction.")
        elif oil_grade == "Standard":
            reasons.append("Standard grade oil supports stable demand in this market.")
        else:
            reasons.append("Demand is high despite budget grade because other market conditions are favorable.")

        if festival_season == 1:
            reasons.append("Festival season is increasing buyer interest for agarwood oil.")
        else:
            reasons.append("Demand remains high even without a festival-season boost.")

        if prediction_month == 12:
            reasons.append("December is a strong seasonal period for agarwood oil sales.")
        elif prediction_week in [3, 4]:
            reasons.append(f"Week {prediction_week} shows stronger weekly demand movement.")
        else:
            reasons.append("The selected time period supports high demand when combined with the other inputs.")

        reasons.append(market_reason)
        return reasons[:4]

    if demand_category == "Medium":
        reasons = []
        if oil_grade == "Premium":
            reasons.append("Premium grade supports demand, but the remaining conditions keep it at a stable level.")
        elif oil_grade == "Standard":
            reasons.append("Standard grade oil supports stable export demand.")
        else:
            reasons.append("Budget grade creates price-sensitive demand, keeping the result in a moderate range.")

        if festival_season == 1:
            reasons.append("Festival season adds buyer interest, but the overall pattern remains medium.")
        else:
            reasons.append("Non-festival timing keeps demand closer to a normal market level.")

        if prediction_week in [3, 4]:
            reasons.append(f"Week {prediction_week} adds some weekly demand support.")
        else:
            reasons.append(f"Week {prediction_week} indicates a more conservative weekly demand pattern.")

        reasons.append(market_reason)
        return reasons[:4]

    reasons = []
    if oil_grade == "Budget":
        reasons.append("Budget grade oil is the main reason demand remains price-sensitive.")
    elif oil_grade == "Standard":
        reasons.append("Standard grade demand is stable, but the selected conditions keep demand low.")
    else:
        reasons.append("Premium grade helps demand, but the selected market pattern still remains low.")

    if festival_season == 1:
        reasons.append("Festival season adds buyer interest, but it is not enough to move this case above low demand.")
    else:
        reasons.append("Non-festival timing limits buyer activity for this prediction.")

    if prediction_month == 12:
        reasons.append("December supports seasonal demand, but the other inputs keep the prediction low.")
    elif prediction_week in [1, 2]:
        reasons.append(f"Week {prediction_week} shows a more conservative weekly demand pattern.")
    else:
        reasons.append("The selected time period does not create enough demand pressure for a higher category.")

    reasons.append(market_reason)
    return reasons[:4]


class DemandPredictor:
    def __init__(self, model_path: Path = MODEL_PATH):
        self.model_path = model_path
        self.model = None

    def _load_model(self):
        if self.model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Demand model not found at {self.model_path}")
            self.model = load(self.model_path)
        return self.model

    def _normalize_legacy_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        if "prediction_month" in payload:
            return payload

        period = str(payload.get("prediction_period", "Next Month")).lower()
        today = date.today()
        if "week" in period:
            month = today.month
            week = min(4, max(1, (today.day - 1) // 7 + 2))
        elif "quarter" in period:
            month = ((today.month + 2 - 1) % 12) + 1
            week = 1
        else:
            month = (today.month % 12) + 1
            week = 1

        normalized = payload.copy()
        normalized["prediction_month"] = month
        normalized["prediction_week"] = week
        return normalized

    def _build_model_input(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload = self._normalize_legacy_payload(payload)
        oil_type = OIL_TYPE_ALIASES.get(str(payload["oil_type"]), str(payload["oil_type"]))
        market_country = str(payload["market_country"])
        market_region = str(payload.get("market_region") or COUNTRY_TO_REGION[market_country])

        return {
            "Oil_Type": oil_type,
            "Oil_Grade": str(payload["oil_grade"]).title(),
            "Market_Region": market_region,
            "Market_Country": market_country,
            "Prediction_month": int(payload["prediction_month"]),
            "Prediction_week": int(payload["prediction_week"]),
            "Festival_Season": 1 if bool(payload["festival_season"]) else 0,
        }

    def predict(
        self,
        payload: dict[str, Any],
        current_date: date | None = None,
    ) -> dict[str, Any]:
        model_input = self._build_model_input(payload)
        input_frame = pd.DataFrame([model_input], columns=MODEL_INPUT_COLUMNS)
        predicted_index = float(self._load_model().predict(input_frame)[0])
        predicted_index = max(0.0, min(100.0, predicted_index))
        predicted_index = round(predicted_index, 2)
        demand_category = demand_category_from_index(predicted_index)
        predicted_time = resolve_prediction_period(
            int(model_input["Prediction_month"]),
            int(model_input["Prediction_week"]),
            current_date=current_date,
        )
        reasons = generate_prediction_reasons(model_input, demand_category)
        price_range = get_recommended_price_range(
            str(model_input["Oil_Type"]),
            predicted_index,
            demand_category,
        )
        export_date = (
            f"{predicted_time['month_name']} {predicted_time['year']}, "
            f"Week {predicted_time['week']}"
        )

        response = {
            "oil_name": model_input["Oil_Type"],
            "oil_grade": model_input["Oil_Grade"],
            "export_country": model_input["Market_Country"],
            "export_date": export_date,
            "predicted_time": predicted_time,
            "demand_index": predicted_index,
            "demand_category": demand_category,
            "reasons": reasons,
            "recommended_price_range": price_range,
            # Backward-compatible aliases for the current chatbot module.
            "market_region": model_input["Market_Region"],
            "market_country": model_input["Market_Country"],
            "festival_season": bool(model_input["Festival_Season"]),
            "demand_level": demand_category,
            "why": reasons,
            "selected_details": {
                "oil_type": model_input["Oil_Type"],
                "oil_grade": model_input["Oil_Grade"],
                "market_country": model_input["Market_Country"],
                "prediction_period": payload.get("prediction_period"),
            },
        }

        return response
