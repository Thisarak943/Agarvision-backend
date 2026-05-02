from pathlib import Path
import json
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parents[1]

IMAGE_MODEL_DIR = BASE_DIR / "model" / "image_model"

IMAGE_MODEL_PATH = IMAGE_MODEL_DIR / "agarwood_3class_model.keras"
CLASS_NAMES_PATH = IMAGE_MODEL_DIR / "agarwood_3class_class_names.json"
RESIN_VALIDATOR_MODEL_PATH = IMAGE_MODEL_DIR / "resin_validator_model.keras"
RESIN_VALIDATOR_CLASS_NAMES_PATH = IMAGE_MODEL_DIR / "resin_validator_class_names.json"

_image_model = None
_class_names = None
_resin_validator_model = None
_resin_validator_class_names = None

def get_image_model():
    global _image_model
    if _image_model is None:
        if not IMAGE_MODEL_PATH.exists():
            raise FileNotFoundError(f"Image model not found: {IMAGE_MODEL_PATH}")
        _image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    return _image_model

def get_class_names():
    global _class_names
    if _class_names is None:
        if not CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(f"Class names not found: {CLASS_NAMES_PATH}")
        _class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return _class_names

def get_resin_validator_model():
    global _resin_validator_model
    if _resin_validator_model is None:
        if not RESIN_VALIDATOR_MODEL_PATH.exists():
            raise FileNotFoundError(f"Resin validator model not found: {RESIN_VALIDATOR_MODEL_PATH}")
        _resin_validator_model = tf.keras.models.load_model(RESIN_VALIDATOR_MODEL_PATH)
    return _resin_validator_model

def get_resin_validator_class_names():
    global _resin_validator_class_names
    if _resin_validator_class_names is None:
        if not RESIN_VALIDATOR_CLASS_NAMES_PATH.exists():
            raise FileNotFoundError(f"Resin validator class names not found: {RESIN_VALIDATOR_CLASS_NAMES_PATH}")
        _resin_validator_class_names = json.loads(
            RESIN_VALIDATOR_CLASS_NAMES_PATH.read_text(encoding="utf-8")
        )
    return _resin_validator_class_names
