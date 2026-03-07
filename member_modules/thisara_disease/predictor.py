import json
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# -------------------------------------------------
# Paths
# -------------------------------------------------
MODEL_PATH = "member_modules/thisara_disease/model/agarwood_leaf_disease_model.keras"
REMEDIES_PATH = "member_modules/thisara_disease/remedies.json"

# -------------------------------------------------
# Class order (CONFIRMED from your notebook)
# -------------------------------------------------
CLASS_NAMES = [
    "Downy mildew",
    "Mealy bugs",
    "Mosaic Viruses",
    "Translucent lesion"
]

# EfficientNetB0 input size (from training)
IMG_SIZE = (224, 224)

_model = None
_remedies = None


# -------------------------------------------------
# Load remedies once
# -------------------------------------------------
def _load_remedies():
    global _remedies
    if _remedies is None:
        if os.path.exists(REMEDIES_PATH):
            with open(REMEDIES_PATH, "r", encoding="utf-8") as f:
                _remedies = json.load(f)
        else:
            _remedies = {}
    return _remedies


# -------------------------------------------------
# Load model once
# -------------------------------------------------
def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


# -------------------------------------------------
# Preprocess image (MATCHES TRAINING EXACTLY)
# -------------------------------------------------
def _preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    arr = np.array(image).astype("float32")
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

    # IMPORTANT: same as training
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)

    return arr


# -------------------------------------------------
# Prediction
# -------------------------------------------------
def predict_pil_image(pil_image: Image.Image) -> dict:
    model = _load_model()
    remedies_map = _load_remedies()

    x = _preprocess(pil_image)
    preds = model.predict(x, verbose=0)[0]

    if len(preds) != len(CLASS_NAMES):
        raise ValueError(
            f"Model outputs {len(preds)} classes but CLASS_NAMES has {len(CLASS_NAMES)}"
        )

    # Top-1 & Top-2 analysis
    sorted_indices = np.argsort(preds)
    top1_idx = int(sorted_indices[-1])
    top2_idx = int(sorted_indices[-2])

    top1_prob = float(preds[top1_idx])
    top2_prob = float(preds[top2_idx])
    gap = top1_prob - top2_prob

    predicted_class = CLASS_NAMES[top1_idx]

    # Probability map (for transparency)
    all_probs = {
        CLASS_NAMES[i]: round(float(preds[i]), 4)
        for i in range(len(CLASS_NAMES))
    }

    # -------------------------------------------------
    # OOD / NON-AGARWOOD REJECTION (DOES NOT HARM REAL CASES)
    # -------------------------------------------------
    # Conditions chosen based on your real outputs:
    # - Agarwood diseases → confidence usually > 0.70
    # - Jack / other leaves → confidence ~0.55–0.60 and mixed probs
    if top1_prob < 0.65 or gap < 0.30:
        disease_out = "Not an Agarwood Leaf"
        remedies = [
            "The uploaded leaf does not belong to the agarwood domain",
            "This model is trained only on agarwood leaf diseases",
            "Please upload a clear agarwood leaf image"
        ]
    else:
        disease_out = predicted_class
        remedies = remedies_map.get(disease_out, [])

    return {
        "predicted_disease": disease_out,
        "confidence": round(top1_prob, 4),
        "remedies": remedies,
        "all_probabilities": all_probs,
        "top2": {
            "top1": {
                "label": CLASS_NAMES[top1_idx],
                "prob": round(top1_prob, 4)
            },
            "top2": {
                "label": CLASS_NAMES[top2_idx],
                "prob": round(top2_prob, 4)
            },
            "gap": round(gap, 4)
        }
    }
