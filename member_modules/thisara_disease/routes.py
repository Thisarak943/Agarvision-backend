from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io

from member_modules.thisara_disease.predictor import predict_pil_image

router = APIRouter()

@router.get("/info")
def module_info():
    return {
        "module": "thisara-disease",
        "endpoints": ["/thisara/predict", "/thisara/info"]
    }

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        result = predict_pil_image(image)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
