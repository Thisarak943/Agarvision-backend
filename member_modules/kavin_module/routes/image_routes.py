from fastapi import APIRouter, UploadFile, File, HTTPException
import numpy as np
import tensorflow as tf
from pathlib import Path
from fastapi.responses import HTMLResponse
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import io

from member_modules.kavin_module.services.image_model_loader import (
    get_image_model,
    get_class_names,
)

from member_modules.kavin_module.model.image_model.gradcam import make_gradcam_overlay

router = APIRouter()

IMG_SIZE = (224, 224)


def preprocess_pil_for_efficientnet(pil_img: Image.Image):
    """Returns a batched tensor (1, 224, 224, 3) preprocessed for EfficientNet."""
    pil_img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(pil_img).astype(np.float32)
    arr = preprocess_input(arr)  # EfficientNet preprocess
    x = tf.expand_dims(arr, axis=0)
    return x


@router.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    img_bytes = await file.read()

    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)

    x = tf.expand_dims(img, axis=0)

    model = get_image_model()
    probs = model.predict(x, verbose=0)[0]

    class_names = get_class_names()
    pred_idx = int(np.argmax(probs))
    pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    return {
        "status": "ok",
        "predicted_class": pred_label,
        "predicted_index": pred_idx,
        "confidence": float(probs[pred_idx]),
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

