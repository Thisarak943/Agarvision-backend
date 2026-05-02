from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pandas as pd

from member_modules.kavin_module.services.model_loader import get_numeric_model_bundle

router = APIRouter()


# ---------- INPUT SCHEMA ----------
class AgarwoodInput(BaseModel):
    Chip_Sample_Weight_g: float
    Drying_Time_Days: int
    Drying_Method: str
    Storage_Type: str
    Storage_Duration_Days: int
    Contamination_Level: str


@router.get("/options")
def get_numeric_options():
    models = get_numeric_model_bundle()
    numeric_options = models["options"]

    return {
        "fields": {
            "Chip_Sample_Weight_g": {"type": "number", "unit": "g"},
            "Drying_Time_Days": {"type": "integer", "unit": "days"},
            "Drying_Method": {"type": "select", "options": numeric_options["Drying_Method"]},
            "Storage_Type": {"type": "select", "options": numeric_options["Storage_Type"]},
            "Storage_Duration_Days": {"type": "integer", "unit": "days"},
            "Contamination_Level": {
                "type": "select",
                "options": numeric_options["Contamination_Level"],
            },
        }
    }


# ---------- PREDICT ----------
@router.post("/predict")
def predict_numeric(data: AgarwoodInput):
    payload = data.model_dump()
    models = get_numeric_model_bundle()
    feature_columns = models["feature_columns"]
    numeric_options = models["options"]

    for field, allowed_values in numeric_options.items():
        value = str(payload[field]).strip().lower()
        if value not in allowed_values:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid {field}: '{payload[field]}'. Allowed values: {allowed_values}",
            )
        payload[field] = value

    model_input = {column: payload[column] for column in feature_columns}
    df = pd.DataFrame([model_input], columns=feature_columns)

    export_readiness = models["export_readiness_model"].predict(df)[0]
    reason = models["reason_model"].predict(df)[0]
    improvement_tip = models["improvement_tip_model"].predict(df)[0]

    # `primary_reason` and `additional_reason` are kept temporarily so older
    # frontend screens do not break while moving to the new single reason field.
    return {
        "export_readiness": export_readiness,
        "reason": reason,
        "primary_reason": reason,
        "additional_reason": "",
        "improvement_tip": improvement_tip,
        "inputs_used": model_input,
    }
