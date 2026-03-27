"""
This file collects the app settings in one place.
Most values come from env vars, with safe defaults as backup.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    from utils.core_config import get_core_config
except ImportError:
    from backend.utils.core_config import get_core_config


BASE_DIR = Path(__file__).resolve().parent.parent
core_config = get_core_config()

MLP_MODEL_PATH = core_config.mlp_model_path
AE_MODEL_PATH = core_config.ae_model_path
SCALER_PATH = core_config.scaler_path
CALIBRATOR_PATH = core_config.calibrator_path
THRESHOLDS_PATH = core_config.thresholds_path

ALT_MLP = core_config.alt_mlp_paths
ALT_AE = core_config.alt_ae_paths
ALT_SCALER = core_config.alt_scaler_paths


def _env_path(name: str, default: Path) -> Path:
    # Turn relative paths into full paths from the backend folder.
    raw_value = os.getenv(name, str(default))
    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (BASE_DIR / candidate).resolve()


SHAP_BG_PATH = _env_path("SHAP_BG_PATH", BASE_DIR / "shap_background.npy")
SHAP_BG_MAX = int(os.getenv("SHAP_BG_MAX", "500"))

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "trustlens_db")
MONGO_TX_COLLECTION = os.getenv("MONGO_TX_COLLECTION", "transactions")
MONGO_REPORT_COLLECTION = os.getenv("MONGO_REPORT_COLLECTION", "reports")
MONGO_USER_COLLECTION = os.getenv("MONGO_USER_COLLECTION", "users")
MONGO_SETTINGS_COLLECTION = os.getenv("MONGO_SETTINGS_COLLECTION", "settings")
MONGO_ACTIVITY_COLLECTION = os.getenv("MONGO_ACTIVITY_COLLECTION", "activity_logs")
MONGO_API_KEYS_COLLECTION = os.getenv("MONGO_API_KEYS_COLLECTION", "api_keys")
MONGO_REQUEST_LOG_COLLECTION = os.getenv("MONGO_REQUEST_LOG_COLLECTION", "request_logs")

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
API_KEY_DEFAULT_EXPIRY_DAYS = int(os.getenv("API_KEY_DEFAULT_EXPIRY_DAYS", "180"))
ADMIN_SETUP_KEY = os.getenv("ADMIN_SETUP_KEY", "change-me-in-env")

LOG_PATH = core_config.trustlens_audit_log_path

ALLOWED_ADMIN_ROLES = {"admin", "analyst"}
ALLOWED_USER_ROLES = {"admin", "analyst", "user"}

CORS_ORIGINS = [
    # Local dev URLs plus the deployed frontend.
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "https://trustlens-frontend-beta.vercel.app",
]
