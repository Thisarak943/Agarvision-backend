import os

# You can change these later if needed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Thisara model path (default)
THISARA_MODEL_PATH = os.getenv(
    "THISARA_MODEL_PATH",
    "member_modules/thisara-disease/model/agarwood_leaf_disease_model.keras"
)

THISARA_REMEDIES_PATH = os.getenv(
    "THISARA_REMEDIES_PATH",
    "member_modules/thisara-disease/remedies.json"
)
