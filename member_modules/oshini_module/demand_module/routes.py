from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .predictor import DemandPredictor, CURRENT_PRICES_LKR

router = APIRouter(prefix="/oshini", tags=["Market Intelligence - Demand"])

predictor = DemandPredictor()


class DemandRequest(BaseModel):
    oil_type: str
    oil_grade: str
    market_region: str
    market_country: str
    prediction_period: str   # "Next Week" | "Next Month" | "Next Quarter"
    festival_season: bool


@router.get("/health")
def health():
    return {"status": "ok", "module": "oshini_module"}


@router.post("/predict-demand")
def predict_demand(req: DemandRequest):
    # Basic validation (so frontend errors are clear)
    if req.oil_type not in CURRENT_PRICES_LKR:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid oil_type '{req.oil_type}'."
        )

    try:
        result = predictor.predict(req.model_dump())
        return result
    except ValueError as e:
        # encoder / invalid category errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    