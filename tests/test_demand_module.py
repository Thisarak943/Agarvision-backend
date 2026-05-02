from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from member_modules.oshini_module.demand_module.predictor import (
    DemandPredictor,
    resolve_prediction_period,
)
from member_modules.oshini_module.demand_module.schemas import DemandRequest


def test_resolve_prediction_period_rolls_months_into_upcoming_12_months():
    current_date = date(2026, 5, 1)

    december = resolve_prediction_period(12, 3, current_date=current_date)
    january = resolve_prediction_period(1, 2, current_date=current_date)

    assert december["year"] == 2026
    assert december["month_name"] == "December"
    assert december["forecast_horizon_months"] == 7
    assert january["year"] == 2027
    assert january["month_name"] == "January"
    assert january["forecast_horizon_months"] == 8


def test_request_accepts_month_name_and_normalizes_oil_alias():
    request = DemandRequest(
        oil_type="Ravana",
        oil_grade="premium",
        market_region="Middle East",
        market_country="UAE",
        prediction_month="January",
        prediction_week=2,
        festival_season=True,
    )

    assert request.oil_type == "Silani Ravana"
    assert request.oil_grade == "Premium"
    assert request.prediction_month == 1


def test_request_rejects_mismatched_region_and_country():
    with pytest.raises(ValidationError):
        DemandRequest(
            oil_type="Silani Ravana",
            oil_grade="Premium",
            market_region="Europe",
            market_country="UAE",
            prediction_month="December",
            prediction_week=3,
            festival_season=True,
        )


def test_predictor_returns_mobile_friendly_response():
    request = DemandRequest(
        oil_type="Silani Ravana",
        oil_grade="Premium",
        market_region="Middle East",
        market_country="UAE",
        prediction_month="December",
        prediction_week=3,
        festival_season=True,
    )

    result = DemandPredictor().predict(
        request.model_dump(),
        current_date=date(2026, 5, 1),
    )

    assert result["oil_name"] == "Silani Ravana"
    assert result["export_country"] == "UAE"
    assert result["export_date"] == "December 2026, Week 3"
    assert result["predicted_time"]["year"] == 2026
    assert result["predicted_time"]["month_name"] == "December"
    assert result["demand_category"] in {"Low", "Medium", "High"}
    assert result["recommended_price_range"]["min_price_lkr"] < result["recommended_price_range"]["max_price_lkr"]
    assert result["reasons"]
    assert "demand_message" not in result
    assert "price_message" not in result
    assert "feature_impacts" not in result
