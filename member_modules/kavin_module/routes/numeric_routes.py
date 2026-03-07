from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel
import pandas as pd

from member_modules.kavin_module.services.model_loader import get_numeric_model

router = APIRouter()



# ---------- INPUT SCHEMA (MATCHES YOUR HTML FORM) ----------
class AgarwoodInput(BaseModel):
    Chip_Sample_Weight_g: float
    Drying_Time_Days: int
    Drying_Method: str
    Storage_Type: str
    Storage_Duration_Weeks: int
    Contamination_Level: str

# ---------- PREDICT ----------
@router.post("/predict")
def predict_numeric(data: AgarwoodInput):
    model = get_numeric_model()

    df = pd.DataFrame([data.model_dump()])

    # Normalize categorical text (same logic you had before)
    for c in ["Drying_Method", "Storage_Type", "Contamination_Level"]:
        df[c] = df[c].astype(str).str.strip().str.lower()

    pred = model.predict(df)[0]

    # Your model output order (same as your standalone)
    return {
        "export_readiness": pred[0],
        "primary_reason": pred[1],
        "additional_reason": pred[2],
        "improvement_tip": pred[3],
    }
