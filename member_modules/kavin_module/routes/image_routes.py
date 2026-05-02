from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import tensorflow as tf
from pathlib import Path
from fastapi.responses import HTMLResponse
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess_input
from PIL import Image
import io

from member_modules.kavin_module.services.image_model_loader import (
    get_image_model,
    get_class_names,
    get_resin_validator_model,
    get_resin_validator_class_names,
)

from member_modules.kavin_module.model.image_model.gradcam import make_gradcam_overlay

router = APIRouter()

IMG_SIZE = (224, 224)
NON_RESIN_REJECTION_CONFIDENCE = 0.80

MARKET_RECOMMENDATIONS = {
    "Premium": {
        "recommended_market": "Middle East luxury export market",
        "market_reason": "Premium resin is best suited for high-value buyers and luxury oud markets.",
    },
    "Tigerwood": {
        "recommended_market": "East Asia and premium craft markets",
        "market_reason": "Tigerwood grade has strong visual quality and fits mid-to-high value resin buyers.",
    },
    "Zebrawood": {
        "recommended_market": "Southeast Asia and local commercial markets",
        "market_reason": "Zebrawood grade is more suitable for broader commercial resin trade.",
    },
}


def preprocess_pil_for_efficientnet(pil_img: Image.Image):
    """Returns a batched tensor (1, 224, 224, 3) preprocessed for EfficientNet."""
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil_img).astype(np.float32)
    arr = preprocess_input(arr)  # EfficientNet preprocess
    x = tf.expand_dims(arr, axis=0)
    return x


def preprocess_bytes_for_efficientnet(img_bytes: bytes):
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return tf.expand_dims(img, axis=0)


def preprocess_bytes_for_convnext(img_bytes: bytes):
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = convnext_preprocess_input(img)
    return tf.expand_dims(img, axis=0)


def get_top_prediction(model, class_names, x):
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)
    confidence = float(probs[pred_idx])
    return pred_idx, pred_label, confidence, probs


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        validator_x = preprocess_bytes_for_convnext(img_bytes)
        grading_x = preprocess_bytes_for_efficientnet(img_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    validator_model = get_resin_validator_model()
    validator_class_names = get_resin_validator_class_names()
    _, validator_label, validator_confidence, _ = get_top_prediction(
        validator_model,
        validator_class_names,
        validator_x,
    )

    if (
        validator_label.strip().lower() == "non resin"
        and validator_confidence >= NON_RESIN_REJECTION_CONFIDENCE
    ):
        return {
            "status": "rejected",
            "message": "The uploaded image does not appear to be agarwood resin. Please upload a clear resin chip image.",
        }

    class_names = get_class_names()
    model = get_image_model()
    pred_idx, pred_label, confidence, _ = get_top_prediction(model, class_names, grading_x)
    market = MARKET_RECOMMENDATIONS.get(pred_label, {})

    return {
        "status": "ok",
        "predicted_class": pred_label,
        "predicted_index": pred_idx,
        "confidence": confidence,
        **market,
    }


@router.post("/gradcam")
async def gradcam_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    img_bytes = await file.read()
    if not img_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # bytes -> PIL
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    model = get_image_model()

    # predict to get class index for gradcam
    x = preprocess_pil_for_efficientnet(pil_img)
    probs = model.predict(x, verbose=0)[0]
    class_index = int(np.argmax(probs))

    # gradcam overlay as base64 png
    gradcam_b64 = make_gradcam_overlay(
        model=model,
        preprocess=preprocess_pil_for_efficientnet,
        pil_img=pil_img,
        class_index=class_index,
    )

    if not gradcam_b64:
        raise HTTPException(status_code=500, detail="Grad-CAM generation failed.")

    return {
        "status": "ok",
        "predicted_index": class_index,
        "gradcam_base64": gradcam_b64,
    }

