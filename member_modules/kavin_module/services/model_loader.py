from pathlib import Path
import joblib

# Base path: .../member_modules/kavin_module
BASE_DIR = Path(__file__).resolve().parents[1]

# ---- Numeric model paths ----
NUMERIC_MODEL_DIR = BASE_DIR / "model" / "numeric_model"
NUMERIC_MODEL_BUNDLE_PATH = NUMERIC_MODEL_DIR / "agarwood_numeric_model_v3.joblib"

_numeric_model = None
_numeric_model_bundle = None

def get_numeric_model():
    """Load legacy numeric model once (lazy load) and reuse."""
    global _numeric_model
    if _numeric_model is None:
        legacy_model_path = NUMERIC_MODEL_DIR / "export_readiness_multioutput_model_v2.joblib"
        if not legacy_model_path.exists():
            raise FileNotFoundError(f"Numeric model not found: {legacy_model_path}")
        _numeric_model = joblib.load(legacy_model_path)
    return _numeric_model

def get_numeric_model_bundle():
    """Load the current Kavin numeric model bundle once and reuse it."""
    global _numeric_model_bundle
    if _numeric_model_bundle is None:
        if not NUMERIC_MODEL_BUNDLE_PATH.exists():
            raise FileNotFoundError(f"Numeric model bundle not found: {NUMERIC_MODEL_BUNDLE_PATH}")
        _numeric_model_bundle = joblib.load(NUMERIC_MODEL_BUNDLE_PATH)
    return _numeric_model_bundle
