import os
import numpy as np
import tensorflow as tf
import joblib
from PIL import Image

# Paths--
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model", "agarwood_model.keras")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "numeric_scaler.pkl")
BARK_VALIDATOR_PATH = os.path.join(BASE_DIR, "model", "bark_validator_final.keras")

_model = None
_label_encoder = None
_scaler = None
_bark_validator = None


def load_all():
    global _model, _label_encoder, _scaler

    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)

    if _label_encoder is None:
        _label_encoder = joblib.load(ENCODER_PATH)

    if _scaler is None:
        _scaler = joblib.load(SCALER_PATH)

    return _model, _label_encoder, _scaler


def load_bark_validator():
    global _bark_validator

    if _bark_validator is None:
        _bark_validator = tf.keras.models.load_model(BARK_VALIDATOR_PATH)

    return _bark_validator


# =========================
# Bark / Not Bark Validation
# =========================
def validate_bark_image(image: Image.Image):
    validator_model = load_bark_validator()

    img = image.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img).astype("float32")
    img_array = np.expand_dims(img_array, axis=0)

    # Same preprocessing used when testing validator model
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    score = float(validator_model.predict(img_array, verbose=0)[0][0])

    # Your trained rule:
    # score < 0.5  => Bark
    # score >= 0.5 => Not_Bark
    is_bark = score < 0.5

    return {
        "is_bark": is_bark,
        "label": "Bark" if is_bark else "Not_Bark",
        "bark_score": score,
    }


# =========================
# Image Preprocessing
# =========================
def preprocess_image(image: Image.Image, target_size=(160, 160)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# =========================
# Numeric Preprocessing (6 features)
# =========================
def preprocess_numeric(
    tree_age: float,
    tree_diameter: float,
    inoculation_count: int,
    months_since_first: float,
    months_since_last: float,
):
    inoculation_duration = months_since_first - months_since_last

    numeric_array = np.array(
        [[
            tree_age,
            tree_diameter,
            inoculation_count,
            months_since_first,
            months_since_last,
            inoculation_duration
        ]],
        dtype="float32"
    )

    _, _, scaler = load_all()
    scaled = scaler.transform(numeric_array)
    return scaled


# =========================
# Predict
# =========================
def predict_stage(
    image: Image.Image,
    tree_age: float,
    tree_diameter: float,
    inoculation_count: int,
    months_since_first: float,
    months_since_last: float,
):
    # First validate image
    bark_check = validate_bark_image(image)

    if not bark_check["is_bark"]:
        return {
            "error": "Please upload a valid agarwood bark image.",
            "bark_validation": bark_check
        }

    # Then run stage prediction
    model, encoder, _ = load_all()

    img_input = preprocess_image(image)
    num_input = preprocess_numeric(
        tree_age,
        tree_diameter,
        inoculation_count,
        months_since_first,
        months_since_last
    )

    preds = model.predict([img_input, num_input], verbose=0)[0]
    predicted_index = int(np.argmax(preds))

    predicted_label = encoder.inverse_transform([predicted_index])[0]
    confidence = float(preds[predicted_index])

    labels = list(encoder.classes_)
    probabilities = {labels[i]: float(preds[i]) for i in range(len(labels))}

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities,
        "bark_validation": bark_check,
        "inputs_used": {
            "tree_age": float(tree_age),
            "tree_diameter": float(tree_diameter),
            "inoculation_count": int(inoculation_count),
            "months_since_first": float(months_since_first),
            "months_since_last": float(months_since_last),
            "inoculation_duration": float(months_since_first - months_since_last),
        }
    }