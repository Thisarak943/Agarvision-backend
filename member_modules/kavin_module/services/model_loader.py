from pathlib import Path
import joblib

# Base path: .../member_modules/kavin_module
BASE_DIR = Path(__file__).resolve().parents[1]

# ---- Numeric model path ----
NUMERIC_MODEL_PATH = BASE_DIR / "model" / "numeric_model" / "export_readiness_multioutput_model_v2.joblib"

_numeric_model = None

def get_numeric_model():
    """Load numeric model once (lazy load) and reuse."""
    global _numeric_model
    if _numeric_model is None:
        if not NUMERIC_MODEL_PATH.exists():
            raise FileNotFoundError(f"Numeric model not found: {NUMERIC_MODEL_PATH}")
        _numeric_model = joblib.load(NUMERIC_MODEL_PATH)
    return _numeric_model
