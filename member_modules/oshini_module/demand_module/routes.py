from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .constants import COUNTRIES_BY_REGION, CURRENT_OIL_PRICES_LKR, MONTH_NUMBER_TO_NAME, OIL_GRADES
from .predictor import DemandPredictor
from .schemas import DemandRequest, DemandResponse

router = APIRouter(prefix="/oshini", tags=["Market Intelligence - Demand"])

predictor = DemandPredictor()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "module": "oshini_module",
        "service": "demand_prediction",
        "model_loaded": predictor.model is not None,
    }


@router.get("/demand-options")
def demand_options():
    return {
        "oil_types": list(CURRENT_OIL_PRICES_LKR.keys()),
        "oil_grades": OIL_GRADES,
        "markets": COUNTRIES_BY_REGION,
        "months": [
            {"value": month, "label": month_name}
            for month, month_name in MONTH_NUMBER_TO_NAME.items()
        ],
        "weeks": [1, 2, 3, 4],
        "festival_season": [False, True],
    }


@router.post("/predict-demand", response_model=DemandResponse)
def predict_demand(req: DemandRequest):
    try:
        return predictor.predict(req.model_dump())
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")
