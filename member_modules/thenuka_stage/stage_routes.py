from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
import io

from member_modules.thenuka_stage.stage_predictor import predict_stage

router = APIRouter()


@router.get("/info")
def info():
    return {
        "module": "Resin Induction Stage Classifier",
        "model": "EfficientNetB0 + MLP Early Fusion",
        "validation": "Bark / Not Bark validator added before stage prediction",
        "inputs": [
            "file (image)",
            "tree_age",
            "tree_diameter",
            "inoculation_count",
            "months_since_first",
            "months_since_last"
        ],
        "note": "Backend computes 6th feature: inoculation_duration = months_since_first - months_since_last"
    }


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    tree_age: float = Form(...),
    tree_diameter: float = Form(...),
    inoculation_count: int = Form(...),
    months_since_first: float = Form(...),
    months_since_last: float = Form(...),
):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        result = predict_stage(
            image=image,
            tree_age=tree_age,
            tree_diameter=tree_diameter,
            inoculation_count=inoculation_count,
            months_since_first=months_since_first,
            months_since_last=months_since_last,
        )

        status_code = 400 if "error" in result else 200
        return JSONResponse(content=result, status_code=status_code)

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )