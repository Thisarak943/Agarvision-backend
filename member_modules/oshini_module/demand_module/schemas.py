from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .constants import (
    COUNTRIES_BY_REGION,
    COUNTRY_TO_REGION,
    CURRENT_OIL_PRICES_LKR,
    MONTH_NAME_TO_NUMBER,
    OIL_GRADES,
    OIL_TYPE_ALIASES,
)


class DemandRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    oil_type: str = Field(..., examples=["Silani Ravana", "Ravana"])
    oil_grade: str = Field(..., examples=["Premium"])
    market_region: str | None = Field(default=None, examples=["Middle East"])
    market_country: str = Field(..., examples=["UAE"])
    prediction_month: str | int = Field(..., examples=["January"])
    prediction_week: int = Field(..., ge=1, le=4, examples=[2])
    festival_season: bool = Field(default=False)

    @field_validator("oil_type")
    @classmethod
    def normalize_oil_type(cls, value: str) -> str:
        normalized = OIL_TYPE_ALIASES.get(value, value)
        if normalized not in CURRENT_OIL_PRICES_LKR:
            allowed = ", ".join(CURRENT_OIL_PRICES_LKR)
            raise ValueError(f"Invalid oil_type '{value}'. Allowed values: {allowed}")
        return normalized

    @field_validator("oil_grade")
    @classmethod
    def validate_oil_grade(cls, value: str) -> str:
        normalized = value.title()
        if normalized not in OIL_GRADES:
            allowed = ", ".join(OIL_GRADES)
            raise ValueError(f"Invalid oil_grade '{value}'. Allowed values: {allowed}")
        return normalized

    @field_validator("prediction_month")
    @classmethod
    def normalize_prediction_month(cls, value: str | int) -> int:
        if isinstance(value, int):
            month_number = value
        else:
            value_text = str(value).strip()
            if value_text.isdigit():
                month_number = int(value_text)
            else:
                month_number = MONTH_NAME_TO_NUMBER.get(value_text.lower())

        if not isinstance(month_number, int) or month_number < 1 or month_number > 12:
            allowed = ", ".join(name.title() for name in MONTH_NAME_TO_NUMBER)
            raise ValueError(f"Invalid prediction_month '{value}'. Allowed values: {allowed}")
        return month_number

    @model_validator(mode="after")
    def validate_market(self) -> "DemandRequest":
        if self.market_country not in COUNTRY_TO_REGION:
            allowed = ", ".join(COUNTRY_TO_REGION)
            raise ValueError(
                f"Invalid market_country '{self.market_country}'. Allowed values: {allowed}"
            )

        expected_region = COUNTRY_TO_REGION[self.market_country]
        if not self.market_region:
            self.market_region = expected_region
        elif self.market_region not in COUNTRIES_BY_REGION:
            allowed = ", ".join(COUNTRIES_BY_REGION)
            raise ValueError(
                f"Invalid market_region '{self.market_region}'. Allowed values: {allowed}"
            )
        elif self.market_region != expected_region:
            raise ValueError(
                f"Invalid market selection: {self.market_country} belongs to "
                f"{expected_region}, not {self.market_region}."
            )

        return self


class PredictedTime(BaseModel):
    year: int
    month: int
    month_name: str
    week: int
    forecast_horizon_months: int
    forecast_horizon: str
    forecast_scope: str


class RecommendedPriceRange(BaseModel):
    current_selling_price_lkr: int
    min_price_lkr: int
    max_price_lkr: int


class DemandResponse(BaseModel):
    oil_name: str
    oil_grade: str
    export_country: str
    export_date: str
    predicted_time: PredictedTime
    demand_index: float
    demand_category: str
    reasons: list[str]
    recommended_price_range: RecommendedPriceRange
