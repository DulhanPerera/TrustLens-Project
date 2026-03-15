"""
TrustLens FastAPI API
(MLP + Autoencoder + optional SHAP + MongoDB + Google Auth + Role-Based Access + .env)

Put these files in the SAME folder as this app.py (backend/):
  - mlp_model.keras
  - autoencoder.keras
  - scaler.joblib
  - isotonic_calibrator.joblib   (optional)
  - thresholds.json              (optional)

OPTIONAL (recommended for more meaningful SHAP):
  - shap_background.npy          (recommended; real NORMAL samples after scaling)

Run:
  pip install -r requirements.txt
  pip install pymongo python-dotenv google-auth
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
from contextlib import asynccontextmanager

# =============================================================================
# Load Environment Variables
# =============================================================================
from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# Third-Party Library Imports
# =============================================================================
import numpy as np
import tensorflow as tf
import joblib

# MongoDB
from pymongo import AsyncMongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId

# Google Auth
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

# =============================================================================
# FastAPI Framework Imports
# =============================================================================
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# Optional: reduce TF log noise
# ---------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# =============================================================================
# File Paths Configuration
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MLP = BASE_DIR / "mlp_model.keras"
DEFAULT_AE = BASE_DIR / "autoencoder.keras"
DEFAULT_SCALER = BASE_DIR / "scaler.joblib"
DEFAULT_CALIBRATOR = BASE_DIR / "isotonic_calibrator.joblib"
DEFAULT_THRESHOLDS = BASE_DIR / "thresholds.json"

MLP_MODEL_PATH = Path(os.getenv("MLP_MODEL_PATH", str(DEFAULT_MLP)))
AE_MODEL_PATH = Path(os.getenv("AE_MODEL_PATH", str(DEFAULT_AE)))
SCALER_PATH = Path(os.getenv("SCALER_PATH", str(DEFAULT_SCALER)))
CALIBRATOR_PATH = Path(os.getenv("CALIBRATOR_PATH", str(DEFAULT_CALIBRATOR)))
THRESHOLDS_PATH = Path(os.getenv("THRESHOLDS_PATH", str(DEFAULT_THRESHOLDS)))

ALT_MLP = [BASE_DIR / "mlp_best.keras", BASE_DIR / "trustlens_model.keras"]
ALT_AE = [BASE_DIR / "ae_best.keras"]
ALT_SCALER = [BASE_DIR / "preprocess_artifacts.pkl"]

# =============================================================================
# SHAP Background Data Configuration
# =============================================================================
SHAP_BG_PATH = Path(os.getenv("SHAP_BG_PATH", str(BASE_DIR / "shap_background.npy")))
SHAP_BG_MAX = int(os.getenv("SHAP_BG_MAX", "500"))

# =============================================================================
# MongoDB Configuration
# =============================================================================
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "trustlens_db")
MONGO_TX_COLLECTION = os.getenv("MONGO_TX_COLLECTION", "transactions")
MONGO_REPORT_COLLECTION = os.getenv("MONGO_REPORT_COLLECTION", "reports")
MONGO_USER_COLLECTION = os.getenv("MONGO_USER_COLLECTION", "users")
MONGO_SETTINGS_COLLECTION = os.getenv("MONGO_SETTINGS_COLLECTION", "settings")
MONGO_ACTIVITY_COLLECTION = os.getenv("MONGO_ACTIVITY_COLLECTION", "activity_logs")

# =============================================================================
# Google Auth Configuration
# =============================================================================
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_PATH = BASE_DIR / "trustlens_audit.log"

logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# =============================================================================
# Global Variables
# =============================================================================
mlp_model: Optional[tf.keras.Model] = None
ae_model: Optional[tf.keras.Model] = None

scaler = None
calibrator = None
thresholds: Dict[str, Any] = {}

shap_explainer = None
feature_names: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

last_init_error: Optional[str] = None

mongo_client: Optional[AsyncMongoClient] = None
mongo_db = None
mongo_tx_collection = None
mongo_report_collection = None
mongo_user_collection = None
mongo_settings_collection = None
mongo_activity_collection = None

# =============================================================================
# Role Configuration
# =============================================================================
ALLOWED_ADMIN_ROLES = {"admin", "analyst"}
ALLOWED_USER_ROLES = {"admin", "analyst", "user"}

# =============================================================================
# Utility Functions
# =============================================================================
def _first_existing(primary: Path, alts: List[Path]) -> Optional[Path]:
    if primary.exists():
        return primary
    for p in alts:
        if p.exists():
            return p
    return None


def _to_prob(pred: np.ndarray) -> float:
    pred = np.array(pred)

    if pred.ndim == 2 and pred.shape[1] == 2:
        return float(pred[0, 1])

    val = float(pred.reshape(-1)[0])
    if val < 0.0 or val > 1.0:
        return float(tf.sigmoid(val).numpy())
    return val


def _ae_recon(ae: tf.keras.Model, x_scaled: np.ndarray) -> np.ndarray:
    return ae.predict(x_scaled, verbose=0)


def _recon_error_mse(x_scaled: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    return np.mean(np.square(x_scaled - x_hat), axis=1)


def _anomaly_score_fallback(re_err: float, thr_re: float) -> float:
    thr_re = float(max(thr_re, 1e-12))
    sigma = 0.20 * thr_re
    z = (re_err - thr_re) / max(sigma, 1e-12)
    return float(1.0 / (1.0 + np.exp(-z)))


def _per_feature_sq_error(x_row: np.ndarray, x_hat_row: np.ndarray) -> np.ndarray:
    return np.square(x_row - x_hat_row).reshape(-1)


def _sanitize_for_mongo(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_mongo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_mongo(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_mongo(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _normalize_role(role: Optional[str]) -> str:
    role = (role or "user").strip().lower()
    if role not in ALLOWED_USER_ROLES:
        return "user"
    return role


async def log_activity(action: str, actor: str = "system", details: Optional[dict] = None):
    if mongo_activity_collection is None:
        return

    doc = {
        "action": action,
        "actor": actor,
        "details": details or {},
        "created_at": datetime.utcnow(),
    }
    await mongo_activity_collection.insert_one(doc)


def _serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    if doc is None:
        return doc

    def convert(value):
        if isinstance(value, ObjectId):
            return str(value)
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, list):
            return [convert(v) for v in value]
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    return convert(doc)

# =============================================================================
# Human-Friendly Explanation Helpers
# =============================================================================
def _feature_label(name: str) -> str:
    if name == "Time":
        return "transaction timing pattern"
    if name == "Amount":
        return "transaction amount pattern"
    if isinstance(name, str) and name.startswith("V"):
        return f"hidden transaction behaviour pattern ({name})"
    return str(name)


def _risk_level(score_pct: float) -> str:
    if score_pct >= 70:
        return "HIGH"
    if score_pct >= 40:
        return "MEDIUM"
    return "LOW"


def _join_labels(items: List[str]) -> str:
    cleaned = [str(x) for x in items if str(x).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"

# =============================================================================
# Pydantic Request Schemas
# =============================================================================
class Transaction(BaseModel):
    transaction_id: Optional[str] = None
    Time: float
    V_features: List[float] = Field(..., min_items=28, max_items=28)
    Amount: float
    include_xai: bool = True
    top_k: int = Field(4, ge=1, le=15)


class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_items=50, max_items=500)
    include_xai: bool = False
    top_k: int = Field(4, ge=1, le=15)


class ReportRequest(BaseModel):
    transaction_id: Optional[str] = None
    Time: float
    V_features: List[float] = Field(..., min_items=28, max_items=28)
    Amount: float
    top_k: int = Field(6, ge=1, le=15)
    include_shap: bool = True
    include_recon: bool = True
    include_raw_features: bool = False


class GoogleAuthRequest(BaseModel):
    credential: str
    login_type: str = "dashboard"


class UpdateUserRoleRequest(BaseModel):
    role: str


class AnalystNoteRequest(BaseModel):
    note: str


class SettingUpdate(BaseModel):
    key: str
    value: Any

# =============================================================================
# Asset Loading
# =============================================================================
def load_assets():
    global mlp_model, ae_model, scaler, calibrator, thresholds
    global shap_explainer, feature_names, last_init_error

    last_init_error = None
    shap_explainer = None

    try:
        mlp_path = _first_existing(MLP_MODEL_PATH, ALT_MLP)
        ae_path = _first_existing(AE_MODEL_PATH, ALT_AE)

        if mlp_path is None:
            raise FileNotFoundError(f"MLP model not found. Tried: {MLP_MODEL_PATH} + {ALT_MLP}")
        if ae_path is None:
            raise FileNotFoundError(f"Autoencoder not found. Tried: {AE_MODEL_PATH} + {ALT_AE}")

        mlp_model = tf.keras.models.load_model(str(mlp_path), compile=False)
        ae_model = tf.keras.models.load_model(str(ae_path), compile=False)

        thresholds = {}
        if THRESHOLDS_PATH.exists():
            with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                thresholds = json.load(f) or {}

        fn = thresholds.get("features")
        if isinstance(fn, list) and len(fn) == 30:
            feature_names = fn
        else:
            feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

        scaler_path = _first_existing(SCALER_PATH, ALT_SCALER)
        if scaler_path is None:
            raise FileNotFoundError(f"Scaler not found. Tried: {SCALER_PATH} + {ALT_SCALER}")

        shap_background_from_artifacts = None

        if scaler_path.name.endswith(".pkl"):
            artifacts = joblib.load(str(scaler_path))
            if not isinstance(artifacts, dict):
                raise ValueError("preprocess_artifacts.pkl must be a dict.")

            time_scaler = artifacts.get("time_scaler")
            amount_scaler = artifacts.get("amount_scaler")
            if time_scaler is None or amount_scaler is None:
                raise ValueError("preprocess_artifacts.pkl missing time_scaler/amount_scaler")

            class _LegacyScalerWrapper:
                def transform(self, X_2d: np.ndarray) -> np.ndarray:
                    X_2d = np.asarray(X_2d, dtype=np.float32)
                    t = time_scaler.transform(X_2d[:, [0]])
                    a = amount_scaler.transform(X_2d[:, [-1]])
                    mid = X_2d[:, 1:29]
                    return np.concatenate([t, mid, a], axis=1).astype(np.float32)

            scaler = _LegacyScalerWrapper()
            shap_background_from_artifacts = artifacts.get("background", None)
        else:
            scaler = joblib.load(str(scaler_path))

        calibrator = joblib.load(str(CALIBRATOR_PATH)) if CALIBRATOR_PATH.exists() else None

        enable_shap = os.getenv("ENABLE_SHAP", "1") == "1"
        if enable_shap:
            try:
                import shap

                bg_source = "zeros"

                if SHAP_BG_PATH.exists():
                    bg = np.load(str(SHAP_BG_PATH)).astype(np.float32)
                    if bg.ndim != 2 or bg.shape[1] != 30:
                        raise ValueError(
                            f"Invalid shap_background.npy shape: {bg.shape} (expected (N,30))"
                        )
                    if bg.shape[0] > SHAP_BG_MAX:
                        bg = bg[:SHAP_BG_MAX]
                    bg_source = f"file:{SHAP_BG_PATH.name}"

                elif shap_background_from_artifacts is not None:
                    bg = np.array(shap_background_from_artifacts, dtype=np.float32)
                    if bg.ndim != 2 or bg.shape[1] != 30:
                        bg = np.zeros((10, 30), dtype=np.float32)
                        bg_source = "zeros(fallback_bad_artifacts_bg)"
                    else:
                        if bg.shape[0] > SHAP_BG_MAX:
                            bg = bg[:SHAP_BG_MAX]
                        bg_source = "artifacts:background"

                else:
                    bg = np.zeros((10, 30), dtype=np.float32)

                shap_explainer = shap.DeepExplainer(mlp_model, bg)
                logging.info(
                    f"SHAP initialized. background_source={bg_source} shape={tuple(bg.shape)}"
                )

            except Exception as e:
                shap_explainer = None
                logging.warning(f"SHAP init failed (API still works): {e}")

        logging.info("Assets loaded successfully.")
        logging.info(f"MLP: {mlp_path}")
        logging.info(f"AE : {ae_path}")
        logging.info(f"Scaler: {scaler_path}")
        logging.info(f"Calibrator loaded: {calibrator is not None}")
        logging.info(f"Thresholds loaded: {bool(thresholds)}")

    except Exception as e:
        mlp_model, ae_model, scaler, calibrator, shap_explainer = None, None, None, None, None
        last_init_error = str(e)
        logging.exception("Initialization Error (full traceback):")

# =============================================================================
# XAI Helpers
# =============================================================================
def _xai_shap(
    x_scaled: np.ndarray,
    x_raw: np.ndarray,
    top_k: int,
    output_is_fraud: bool,
) -> List[Dict[str, Any]]:
    if shap_explainer is None:
        return []

    try:
        shap_values = shap_explainer.shap_values(x_scaled)

        if isinstance(shap_values, list):
            vals = np.array(shap_values[0]).reshape(-1)
        else:
            vals = np.array(shap_values).reshape(-1)

        if not output_is_fraud:
            vals = -vals

        abs_vals = np.abs(vals)

        fn = feature_names if len(feature_names) == len(vals) else (
            ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        )
        x_raw_1d = np.array(x_raw, dtype=np.float32).reshape(-1)
        x_scaled_1d = np.array(x_scaled, dtype=np.float32).reshape(-1)

        idx = np.argsort(abs_vals)[-top_k:][::-1]

        out: List[Dict[str, Any]] = []
        for i in idx:
            s = float(vals[i])
            out.append(
                {
                    "name": fn[i],
                    "impact": float(abs_vals[i]),
                    "signed": s,
                    "direction": "increase" if s > 0 else "decrease",
                    "value_raw": float(x_raw_1d[i]),
                    "value_scaled": float(x_scaled_1d[i]),
                }
            )
        return out

    except Exception as e:
        logging.warning(f"SHAP compute failed: {e}")
        return []


def _xai_recon_pf(x_scaled_row: np.ndarray, x_hat_row: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
    errs = _per_feature_sq_error(x_scaled_row, x_hat_row)
    fn = feature_names if len(feature_names) == len(errs) else (
        ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    )
    idx = np.argsort(errs)[-top_k:][::-1]
    return [{"name": fn[i], "impact": float(errs[i])} for i in idx]


def _build_text_reason(
    shap_top: List[Dict[str, Any]],
    recon_top: List[Dict[str, Any]],
    time_value: Optional[float] = None,
    amount_value: Optional[float] = None,
    risk_score_pct: Optional[float] = None,
) -> str:
    sections: List[str] = []

    if time_value is not None or amount_value is not None:
        t = f"{float(time_value):.0f}" if time_value is not None else "-"
        a = f"${float(amount_value):.2f}" if amount_value is not None else "-"
        sections.append(f"Transaction analysed with time feature value {t} and amount {a}.")

    sections.append(
        "The system compared this transaction against learned normal and suspicious behaviour patterns."
    )

    if shap_top:
        inc = [_feature_label(d["name"]) for d in shap_top if float(d.get("signed", 0.0)) > 0][:2]
        dec = [_feature_label(d["name"]) for d in shap_top if float(d.get("signed", 0.0)) < 0][:2]

        if inc:
            sections.append(
                "The fraud score increased mainly because of unusual behaviour in "
                + _join_labels(inc)
                + "."
            )

        if dec:
            sections.append(
                "Some behaviour patterns looked closer to normal activity, which slightly reduced the fraud score, especially "
                + _join_labels(dec)
                + "."
            )

    if recon_top:
        drivers = [_feature_label(d["name"]) for d in recon_top[:2]]
        if drivers:
            sections.append(
                "The anomaly model also detected abnormal behaviour in "
                + _join_labels(drivers)
                + "."
            )

    if risk_score_pct is not None:
        sections.append(
            f"Final risk level: {_risk_level(float(risk_score_pct))} ({float(risk_score_pct):.2f}%)."
        )

    return " ".join(sections)

# =============================================================================
# MongoDB Save Helpers
# =============================================================================
async def _save_prediction_to_mongo(tx: Transaction, response_payload: Dict[str, Any]) -> Optional[str]:
    if mongo_tx_collection is None:
        return None

    doc = {
        "transaction_id": tx.transaction_id,
        "type": "predict",
        "input": {
            "Time": float(tx.Time),
            "V_features": [float(v) for v in tx.V_features],
            "Amount": float(tx.Amount),
        },
        "result": _sanitize_for_mongo(response_payload),
        "status": response_payload.get("status"),
        "is_fraud": bool(response_payload.get("is_fraud", False)),
        "risk_score": float(response_payload.get("risk_score", 0.0)),
        "mlp_prob_raw": float(response_payload.get("mlp_prob_raw", 0.0)),
        "fraud_prob": float(response_payload.get("fraud_prob", 0.0)),
        "recon_error": float(response_payload.get("recon_error", 0.0)),
        "anomaly_score": float(response_payload.get("anomaly_score", 0.0)),
        "combined_score": float(response_payload.get("combined_score", 0.0)),
        "thresholds": _sanitize_for_mongo(response_payload.get("thresholds", {})),
        "explanation": response_payload.get("explanation"),
        "xai_data": _sanitize_for_mongo(response_payload.get("xai_data", [])),
        "ae_xai_data": _sanitize_for_mongo(response_payload.get("ae_xai_data", [])),
        "analyst_notes": [],
        "analyst_override": False,
        "created_at": datetime.utcnow(),
    }

    result = await mongo_tx_collection.insert_one(doc)
    return str(result.inserted_id)


async def _save_report_to_mongo(req: ReportRequest, report_payload: Dict[str, Any]) -> Optional[str]:
    if mongo_report_collection is None:
        return None

    doc = {
        "transaction_id": req.transaction_id,
        "type": "report",
        "input": {
            "Time": float(req.Time),
            "V_features": [float(v) for v in req.V_features],
            "Amount": float(req.Amount),
        },
        "decision": report_payload.get("decision"),
        "risk_score": float(report_payload.get("risk_score", 0.0)),
        "confidence": float(report_payload.get("confidence", 0.0)),
        "signals": _sanitize_for_mongo(report_payload.get("signals", {})),
        "thresholds": _sanitize_for_mongo(report_payload.get("thresholds", {})),
        "explanations": _sanitize_for_mongo(report_payload.get("explanations", {})),
        "meta": _sanitize_for_mongo(report_payload.get("meta", {})),
        "created_at": datetime.utcnow(),
    }

    result = await mongo_report_collection.insert_one(doc)
    return str(result.inserted_id)


async def _save_batch_to_mongo(batch_payload: Dict[str, Any]) -> Optional[str]:
    if mongo_tx_collection is None:
        return None

    doc = {
        "type": "predict_batch",
        "count": int(batch_payload.get("count", 0)),
        "sorted_by": batch_payload.get("sorted_by"),
        "thresholds": _sanitize_for_mongo(batch_payload.get("thresholds", {})),
        "results": _sanitize_for_mongo(batch_payload.get("results", [])),
        "created_at": datetime.utcnow(),
    }

    result = await mongo_tx_collection.insert_one(doc)
    return str(result.inserted_id)

# =============================================================================
# FastAPI Lifespan
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mongo_client, mongo_db, mongo_tx_collection, mongo_report_collection
    global mongo_user_collection, mongo_settings_collection, mongo_activity_collection

    try:
        mongo_client = AsyncMongoClient(
            MONGO_URI,
            server_api=ServerApi("1"),
        )
        mongo_db = mongo_client[MONGO_DB_NAME]
        mongo_tx_collection = mongo_db[MONGO_TX_COLLECTION]
        mongo_report_collection = mongo_db[MONGO_REPORT_COLLECTION]
        mongo_user_collection = mongo_db[MONGO_USER_COLLECTION]
        mongo_settings_collection = mongo_db[MONGO_SETTINGS_COLLECTION]
        mongo_activity_collection = mongo_db[MONGO_ACTIVITY_COLLECTION]

        await mongo_client.admin.command("ping")

        await mongo_tx_collection.create_index("created_at")
        await mongo_tx_collection.create_index("status")
        await mongo_tx_collection.create_index("is_fraud")
        await mongo_tx_collection.create_index("transaction_id")

        await mongo_report_collection.create_index("created_at")
        await mongo_report_collection.create_index("decision")
        await mongo_report_collection.create_index("transaction_id")

        await mongo_user_collection.create_index("google_sub", unique=True)
        await mongo_user_collection.create_index("email")

        await mongo_activity_collection.create_index("created_at")
        await mongo_activity_collection.create_index("action")
        await mongo_settings_collection.create_index("key", unique=True)

        logging.info("MongoDB connected successfully.")

    except Exception as e:
        logging.exception(f"MongoDB initialization failed: {e}")
        mongo_client = None
        mongo_db = None
        mongo_tx_collection = None
        mongo_report_collection = None
        mongo_user_collection = None
        mongo_settings_collection = None
        mongo_activity_collection = None

    load_assets()

    yield

    if mongo_client is not None:
        await mongo_client.close()
        logging.info("MongoDB connection closed.")

# =============================================================================
# FastAPI Application Initialization
# =============================================================================
app = FastAPI(
    title="TrustLens API",
    description="Fraud Detection with MLP + Autoencoder + XAI (Explainable AI) + MongoDB + Google Auth",
    lifespan=lifespan,
)

# =============================================================================
# CORS Middleware
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://trustlens-frontend-beta.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Google Authentication Endpoints
# =============================================================================
@app.post("/auth/google")
async def auth_google(payload: GoogleAuthRequest):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID is not configured")

    if mongo_user_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB user collection is not available")

    try:
        login_type = (payload.login_type or "dashboard").strip().lower()
        if login_type not in {"dashboard", "admin"}:
            raise HTTPException(status_code=400, detail="Invalid login_type")

        idinfo = id_token.verify_oauth2_token(
            payload.credential,
            google_requests.Request(),
            GOOGLE_CLIENT_ID,
        )

        google_sub = idinfo.get("sub")
        email = idinfo.get("email")
        name = idinfo.get("name")
        picture = idinfo.get("picture")
        email_verified = bool(idinfo.get("email_verified", False))
        given_name = idinfo.get("given_name")
        family_name = idinfo.get("family_name")

        if not google_sub or not email:
            raise HTTPException(status_code=400, detail="Invalid Google token payload")

        existing_user = await mongo_user_collection.find_one({"google_sub": google_sub})

        if existing_user:
            role = _normalize_role(existing_user.get("role"))
        else:
            role = "user"

        if login_type == "admin" and role not in ALLOWED_ADMIN_ROLES:
            raise HTTPException(
                status_code=403,
                detail="Access denied. Only admins and analysts can access the admin panel."
            )

        user_doc = {
            "google_sub": google_sub,
            "email": email,
            "name": name,
            "picture": picture,
            "given_name": given_name,
            "family_name": family_name,
            "email_verified": email_verified,
            "role": role,
            "last_login_at": datetime.utcnow(),
            "auth_provider": "google",
        }

        await mongo_user_collection.update_one(
            {"google_sub": google_sub},
            {"$set": user_doc},
            upsert=True,
        )

        await log_activity(
            action="google_login",
            actor=email,
            details={"google_sub": google_sub, "role": role, "login_type": login_type},
        )

        logging.info(
            f"google_auth_success: email={email} google_sub={google_sub} role={role} login_type={login_type}"
        )

        return {
            "message": "Google authentication successful",
            "login_type": login_type,
            "user": {
                "google_sub": google_sub,
                "email": email,
                "name": name,
                "picture": picture,
                "given_name": given_name,
                "family_name": family_name,
                "email_verified": email_verified,
                "role": role,
                "login_type": login_type,
            },
        }

    except HTTPException:
        raise
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid Google ID token")
    except Exception as e:
        logging.exception(f"Google auth failed: {e}")
        raise HTTPException(status_code=500, detail=f"Google auth failed: {str(e)}")


@app.get("/users")
async def get_users(limit: int = 20):
    if mongo_user_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB user collection is not available")

    limit = max(1, min(limit, 200))
    cursor = mongo_user_collection.find().sort("last_login_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)

    docs = [_serialize_doc(d) for d in docs]
    return {"count": len(docs), "items": docs}


@app.patch("/users/{user_id}/role")
async def update_user_role(user_id: str, payload: UpdateUserRoleRequest):
    if mongo_user_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB user collection is not available")

    new_role = _normalize_role(payload.role)

    try:
        result = await mongo_user_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"role": new_role}}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    updated_user = await mongo_user_collection.find_one({"_id": ObjectId(user_id)})
    updated_user = _serialize_doc(updated_user)

    await log_activity(
        action="user_role_updated",
        actor="admin",
        details={"user_id": user_id, "new_role": new_role},
    )

    logging.info(f"user_role_updated: user_id={user_id} new_role={new_role}")

    return {
        "message": "User role updated successfully",
        "user": updated_user,
    }

# =============================================================================
# API Endpoint: Single Transaction Prediction
# =============================================================================
@app.post("/predict")
async def predict(tx: Transaction, background_tasks: BackgroundTasks):
    if mlp_model is None or ae_model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready: {last_init_error or 'unknown error'}"
        )

    try:
        x_raw = np.array([[tx.Time] + tx.V_features + [tx.Amount]], dtype=np.float32)
        if x_raw.shape != (1, 30):
            raise ValueError("Input must be 30 features: Time + 28 V-features + Amount")

        x_scaled = scaler.transform(x_raw).astype(np.float32)

        try:
            input_name = mlp_model.input_names[0]
            raw_pred = mlp_model.predict({input_name: x_scaled}, verbose=0)
        except Exception:
            raw_pred = mlp_model.predict(x_scaled, verbose=0)

        p_raw = _to_prob(raw_pred)

        output_is_fraud = os.getenv("MODEL_OUTPUT_IS_FRAUD", "1") == "1"
        p_fraud = p_raw if output_is_fraud else (1.0 - p_raw)

        if calibrator is not None:
            p_fraud = float(calibrator.transform([p_fraud])[0])
        else:
            p_fraud = float(p_fraud)

        x_hat = _ae_recon(ae_model, x_scaled)
        re_err = float(_recon_error_mse(x_scaled, x_hat)[0])

        thr_re = float(thresholds.get("ae_thr_val_norm_99.5pct", os.getenv("AE_THR", "0.01")))
        anomaly_score = _anomaly_score_fallback(re_err, thr_re)

        w = thresholds.get("weights", {"w_mlp": 0.6, "w_ae": 0.4})
        w_mlp = float(w.get("w_mlp", 0.6))
        w_ae = float(w.get("w_ae", 0.4))
        combined_score = float(w_mlp * p_fraud + w_ae * anomaly_score)

        combined_thr = thresholds.get("combined_thr_val_maxf1", {}).get("thr", None)
        combined_thr = (
            float(combined_thr)
            if combined_thr is not None
            else float(os.getenv("COMBINED_THR", "0.5"))
        )

        is_fraud = combined_score >= combined_thr

        top_k = int(tx.top_k)
        shap_data = (
            _xai_shap(x_scaled, x_raw, top_k=top_k, output_is_fraud=output_is_fraud)
            if tx.include_xai else []
        )
        recon_data = _xai_recon_pf(x_scaled[0], x_hat[0], top_k) if tx.include_xai else []

        risk_score_pct = combined_score * 100.0
        explanation = (
            _build_text_reason(
                shap_data,
                recon_data,
                time_value=tx.Time,
                amount_value=tx.Amount,
                risk_score_pct=risk_score_pct,
            )
            if tx.include_xai
            else (
                f"Transaction analysed with time feature value {tx.Time:.0f} "
                f"and amount ${tx.Amount:.2f}. Final risk level: "
                f"{_risk_level(risk_score_pct)} ({risk_score_pct:.2f}%)."
            )
        )

        if is_fraud:
            background_tasks.add_task(
                logging.info,
                f"Fraud Alert: combined={combined_score:.6f}, p_fraud={p_fraud:.6f}, re={re_err:.6f}",
            )

        logging.info(
            f"predict: tx_id={tx.transaction_id} p_raw={p_raw:.6f} p_fraud={p_fraud:.6f} "
            f"re={re_err:.6f} anom={anomaly_score:.6f} combined={combined_score:.6f} is_fraud={is_fraud}"
        )

        response = {
            "transaction_id": tx.transaction_id,
            "is_fraud": bool(is_fraud),
            "status": "BLOCKED" if is_fraud else "APPROVED",
            "risk_score": round(risk_score_pct, 2),
            "mlp_prob_raw": float(p_raw),
            "fraud_prob": float(p_fraud),
            "recon_error": float(re_err),
            "anomaly_score": float(anomaly_score),
            "combined_score": float(combined_score),
            "thresholds": {
                "combined_thr": combined_thr,
                "ae_thr": thr_re,
                "weights": {"w_mlp": w_mlp, "w_ae": w_ae},
                "model_output_is_fraud": output_is_fraud,
            },
            "explanation": explanation,
            "xai_data": shap_data,
            "ae_xai_data": recon_data,
        }

        mongo_id = await _save_prediction_to_mongo(tx, response)
        response["mongo_id"] = mongo_id

        await log_activity(
            action="transaction_predicted",
            actor="system",
            details={
                "transaction_id": tx.transaction_id,
                "mongo_id": mongo_id,
                "status": response["status"],
                "risk_score": response["risk_score"],
            },
        )

        return response

    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal analysis failure: {str(e)}")

# =============================================================================
# API Endpoint: Batch Transaction Prediction
# =============================================================================
@app.post("/predict_batch")
async def predict_batch(req: BatchRequest):
    if mlp_model is None or ae_model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready: {last_init_error or 'unknown error'}"
        )

    try:
        txs = req.transactions
        do_xai = bool(req.include_xai)
        top_k = int(req.top_k)

        X_raw = np.array([[t.Time] + t.V_features + [t.Amount] for t in txs], dtype=np.float32)
        if X_raw.ndim != 2 or X_raw.shape[1] != 30:
            raise ValueError("Each transaction must contain 30 features: Time + 28 V + Amount")

        X_scaled = scaler.transform(X_raw).astype(np.float32)

        try:
            input_name = mlp_model.input_names[0]
            raw_pred = mlp_model.predict({input_name: X_scaled}, verbose=0)
        except Exception:
            raw_pred = mlp_model.predict(X_scaled, verbose=0)

        raw_pred = np.array(raw_pred)

        if raw_pred.ndim == 2 and raw_pred.shape[1] == 2:
            p_raw = raw_pred[:, 1].astype(np.float32)
        else:
            p_raw = raw_pred.reshape(-1).astype(np.float32)
            p_raw = np.where(
                (p_raw < 0.0) | (p_raw > 1.0),
                1.0 / (1.0 + np.exp(-p_raw)),
                p_raw
            ).astype(np.float32)

        output_is_fraud = os.getenv("MODEL_OUTPUT_IS_FRAUD", "1") == "1"
        p_fraud = p_raw if output_is_fraud else (1.0 - p_raw)

        if calibrator is not None:
            p_fraud = calibrator.transform(p_fraud.tolist()).astype(np.float32)

        X_hat = ae_model.predict(X_scaled, verbose=0)
        re_err = np.mean(np.square(X_scaled - X_hat), axis=1).astype(np.float32)

        thr_re = float(thresholds.get("ae_thr_val_norm_99.5pct", os.getenv("AE_THR", "0.01")))
        thr_safe = max(thr_re, 1e-12)
        sigma = 0.20 * thr_safe
        z = (re_err - thr_safe) / max(sigma, 1e-12)
        anomaly_score = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

        w = thresholds.get("weights", {"w_mlp": 0.6, "w_ae": 0.4})
        w_mlp = float(w.get("w_mlp", 0.6))
        w_ae = float(w.get("w_ae", 0.4))
        combined = (w_mlp * p_fraud + w_ae * anomaly_score).astype(np.float32)

        combined_thr = thresholds.get("combined_thr_val_maxf1", {}).get("thr", None)
        combined_thr = (
            float(combined_thr)
            if combined_thr is not None
            else float(os.getenv("COMBINED_THR", "0.5"))
        )

        is_fraud = combined >= combined_thr

        results: List[Dict[str, Any]] = []
        for i in range(len(txs)):
            risk_score_pct = float(combined[i]) * 100.0

            item: Dict[str, Any] = {
                "index": i,
                "transaction_id": txs[i].transaction_id,
                "is_fraud": bool(is_fraud[i]),
                "status": "BLOCKED" if bool(is_fraud[i]) else "APPROVED",
                "risk_score": float(round(risk_score_pct, 2)),
                "fraud_prob": float(p_fraud[i]),
                "recon_error": float(re_err[i]),
                "anomaly_score": float(anomaly_score[i]),
                "combined_score": float(combined[i]),
            }

            if do_xai:
                shap_data = _xai_shap(
                    X_scaled[i:i + 1],
                    X_raw[i:i + 1],
                    top_k=top_k,
                    output_is_fraud=output_is_fraud,
                )
                recon_data = _xai_recon_pf(X_scaled[i], X_hat[i], top_k=top_k)
                item["xai_data"] = shap_data
                item["ae_xai_data"] = recon_data
                item["explanation"] = _build_text_reason(
                    shap_data,
                    recon_data,
                    time_value=float(txs[i].Time),
                    amount_value=float(txs[i].Amount),
                    risk_score_pct=risk_score_pct,
                )

            results.append(item)

        results.sort(key=lambda x: x["risk_score"], reverse=True)

        batch_response = {
            "count": len(results),
            "sorted_by": "risk_score_desc",
            "thresholds": {
                "combined_thr": combined_thr,
                "ae_thr": thr_re,
                "weights": {"w_mlp": w_mlp, "w_ae": w_ae},
                "model_output_is_fraud": output_is_fraud,
            },
            "results": results,
        }

        mongo_id = await _save_batch_to_mongo(batch_response)
        batch_response["mongo_id"] = mongo_id

        await log_activity(
            action="batch_prediction",
            actor="system",
            details={"count": len(results), "mongo_id": mongo_id},
        )

        return batch_response

    except Exception as e:
        logging.error(f"Batch Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

# =============================================================================
# API Endpoint: Detailed Fraud Analysis Report
# =============================================================================
@app.post("/report")
async def report(req: ReportRequest):
    if mlp_model is None or ae_model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not ready: {last_init_error or 'unknown error'}"
        )

    try:
        x_raw = np.array([[req.Time] + req.V_features + [req.Amount]], dtype=np.float32)
        if x_raw.shape != (1, 30):
            raise ValueError("Input must be 30 features: Time + 28 V-features + Amount")

        x_scaled = scaler.transform(x_raw).astype(np.float32)

        try:
            input_name = mlp_model.input_names[0]
            raw_pred = mlp_model.predict({input_name: x_scaled}, verbose=0)
        except Exception:
            raw_pred = mlp_model.predict(x_scaled, verbose=0)

        p_raw = _to_prob(raw_pred)

        output_is_fraud = os.getenv("MODEL_OUTPUT_IS_FRAUD", "1") == "1"
        p_fraud = p_raw if output_is_fraud else (1.0 - p_raw)

        if calibrator is not None:
            p_fraud_cal = float(calibrator.transform([p_fraud])[0])
        else:
            p_fraud_cal = float(p_fraud)

        x_hat = _ae_recon(ae_model, x_scaled)
        re_err = float(_recon_error_mse(x_scaled, x_hat)[0])

        thr_re = float(thresholds.get("ae_thr_val_norm_99.5pct", os.getenv("AE_THR", "0.01")))
        anomaly_score = _anomaly_score_fallback(re_err, thr_re)

        w = thresholds.get("weights", {"w_mlp": 0.6, "w_ae": 0.4})
        w_mlp = float(w.get("w_mlp", 0.6))
        w_ae = float(w.get("w_ae", 0.4))
        combined_score = float(w_mlp * p_fraud_cal + w_ae * anomaly_score)

        combined_thr = thresholds.get("combined_thr_val_maxf1", {}).get("thr", None)
        combined_thr = (
            float(combined_thr)
            if combined_thr is not None
            else float(os.getenv("COMBINED_THR", "0.5"))
        )

        review_margin = float(os.getenv("REVIEW_MARGIN", "0.05"))
        if combined_score >= combined_thr:
            decision = "BLOCKED"
        elif combined_score >= (combined_thr - review_margin):
            decision = "REVIEW"
        else:
            decision = "APPROVED"

        top_k = int(req.top_k)

        shap_data: List[Dict[str, Any]] = []
        if req.include_shap and shap_explainer is not None:
            shap_data = _xai_shap(
                x_scaled,
                x_raw,
                top_k=top_k,
                output_is_fraud=output_is_fraud,
            )

        recon_data: List[Dict[str, Any]] = []
        if req.include_recon:
            recon_data = _xai_recon_pf(x_scaled[0], x_hat[0], top_k)

        summary = _build_text_reason(
            shap_data,
            recon_data,
            time_value=req.Time,
            amount_value=req.Amount,
            risk_score_pct=combined_score * 100.0,
        )

        dist = abs(combined_score - combined_thr)
        confidence = float(min(1.0, dist / max(combined_thr, 1e-6)))

        card: Dict[str, Any] = {
            "transaction_id": req.transaction_id,
            "decision": decision,
            "risk_score": round(combined_score * 100.0, 2),
            "confidence": round(confidence, 3),
            "signals": {
                "mlp_prob_raw": float(p_raw),
                "fraud_prob": float(p_fraud_cal),
                "recon_error": float(re_err),
                "anomaly_score": float(anomaly_score),
                "combined_score": float(combined_score),
            },
            "thresholds": {
                "combined_thr": float(combined_thr),
                "ae_thr": float(thr_re),
                "weights": {"w_mlp": w_mlp, "w_ae": w_ae},
                "model_output_is_fraud": output_is_fraud,
                "review_margin": review_margin,
            },
            "explanations": {
                "summary": summary,
                "shap_top": shap_data,
                "recon_top": recon_data,
            },
            "meta": {
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "models": {
                    "mlp_output_shape": getattr(mlp_model, "output_shape", None),
                    "ae_output_shape": getattr(ae_model, "output_shape", None),
                },
                "xai": {
                    "shap_available": shap_explainer is not None,
                    "feature_names_len": len(feature_names),
                    "shap_bg_path": str(SHAP_BG_PATH),
                    "shap_bg_max": SHAP_BG_MAX,
                },
            },
        }

        if req.include_raw_features:
            card["input"] = {
                "Time": float(req.Time),
                "V_features": [float(x) for x in req.V_features],
                "Amount": float(req.Amount),
            }

        mongo_id = await _save_report_to_mongo(req, card)
        card["mongo_id"] = mongo_id

        await log_activity(
            action="report_generated",
            actor="system",
            details={
                "transaction_id": req.transaction_id,
                "mongo_id": mongo_id,
                "decision": decision,
            },
        )

        return card

    except Exception as e:
        logging.error(f"Report Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

# =============================================================================
# Admin Workflow Endpoints
# =============================================================================
@app.patch("/transactions/{tx_id}/mark-legitimate")
async def mark_legitimate(tx_id: str):
    if mongo_tx_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    try:
        result = await mongo_tx_collection.update_one(
            {"_id": ObjectId(tx_id)},
            {"$set": {
                "status": "APPROVED",
                "is_fraud": False,
                "analyst_override": True,
            }}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid transaction id")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Transaction not found")

    updated_doc = await mongo_tx_collection.find_one({"_id": ObjectId(tx_id)})
    updated_doc = _serialize_doc(updated_doc)

    await log_activity("mark_legitimate", "analyst", {"transaction_id": tx_id})

    return {
        "message": "Transaction marked as legitimate",
        "item": updated_doc,
    }


@app.post("/transactions/{tx_id}/note")
async def add_analyst_note(tx_id: str, payload: AnalystNoteRequest):
    if mongo_tx_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    note_doc = {
        "note": payload.note,
        "created_at": datetime.utcnow(),
    }

    try:
        result = await mongo_tx_collection.update_one(
            {"_id": ObjectId(tx_id)},
            {"$push": {"analyst_notes": note_doc}}
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid transaction id")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Transaction not found")

    updated_doc = await mongo_tx_collection.find_one({"_id": ObjectId(tx_id)})
    updated_doc = _serialize_doc(updated_doc)

    await log_activity(
        "analyst_note_added",
        "analyst",
        {"transaction_id": tx_id, "note": payload.note},
    )

    return {
        "message": "Note added",
        "item": updated_doc,
    }


@app.get("/settings")
async def get_settings():
    if mongo_settings_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    cursor = mongo_settings_collection.find()
    docs = await cursor.to_list(length=100)
    docs = [_serialize_doc(d) for d in docs]

    return {"items": docs}


@app.post("/settings")
async def update_setting(payload: SettingUpdate):
    if mongo_settings_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    await mongo_settings_collection.update_one(
        {"key": payload.key},
        {"$set": {"key": payload.key, "value": payload.value}},
        upsert=True,
    )

    await log_activity(
        "settings_updated",
        "admin",
        {"key": payload.key, "value": payload.value},
    )

    return {"message": "Setting updated"}


@app.get("/activity-logs")
async def get_activity_logs(limit: int = 50):
    if mongo_activity_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    limit = max(1, min(limit, 200))
    cursor = mongo_activity_collection.find().sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    docs = [_serialize_doc(d) for d in docs]

    return {"count": len(docs), "items": docs}

# =============================================================================
# API Endpoint: Health Check
# =============================================================================
@app.get("/health")
def health():
    return {
        "mlp_loaded": mlp_model is not None,
        "ae_loaded": ae_model is not None,
        "scaler_loaded": scaler is not None,
        "calibrator_loaded": calibrator is not None,
        "shap_loaded": shap_explainer is not None,
        "mongodb": {
            "connected": mongo_client is not None and mongo_db is not None,
            "db_name": MONGO_DB_NAME,
            "tx_collection": MONGO_TX_COLLECTION,
            "report_collection": MONGO_REPORT_COLLECTION,
            "user_collection": MONGO_USER_COLLECTION,
            "settings_collection": MONGO_SETTINGS_COLLECTION,
            "activity_collection": MONGO_ACTIVITY_COLLECTION,
        },
        "paths": {
            "base_dir": str(BASE_DIR),
            "mlp_model_path": str(_first_existing(MLP_MODEL_PATH, ALT_MLP) or MLP_MODEL_PATH),
            "ae_model_path": str(_first_existing(AE_MODEL_PATH, ALT_AE) or AE_MODEL_PATH),
            "scaler_path": str(_first_existing(SCALER_PATH, ALT_SCALER) or SCALER_PATH),
            "calibrator_path": str(CALIBRATOR_PATH),
            "thresholds_path": str(THRESHOLDS_PATH),
            "log_path": str(LOG_PATH),
            "shap_bg_path": str(SHAP_BG_PATH),
        },
        "model_info": {
            "mlp_output_shape": getattr(mlp_model, "output_shape", None) if mlp_model else None,
            "mlp_input_names": getattr(mlp_model, "input_names", None) if mlp_model else None,
            "ae_output_shape": getattr(ae_model, "output_shape", None) if ae_model else None,
        },
        "env": {
            "ENABLE_SHAP": os.getenv("ENABLE_SHAP", "1"),
            "MODEL_OUTPUT_IS_FRAUD": os.getenv("MODEL_OUTPUT_IS_FRAUD", "1"),
            "COMBINED_THR": os.getenv("COMBINED_THR", "(using thresholds.json if present)"),
            "AE_THR": os.getenv("AE_THR", "(using thresholds.json if present)"),
            "REVIEW_MARGIN": os.getenv("REVIEW_MARGIN", "0.05"),
            "SHAP_BG_PATH": os.getenv("SHAP_BG_PATH", str(SHAP_BG_PATH)),
            "SHAP_BG_MAX": os.getenv("SHAP_BG_MAX", str(SHAP_BG_MAX)),
            "MONGO_URI_SET": bool(os.getenv("MONGO_URI")),
            "MONGO_DB_NAME": MONGO_DB_NAME,
            "GOOGLE_CLIENT_ID_SET": bool(GOOGLE_CLIENT_ID),
        },
        "last_init_error": last_init_error,
    }

# =============================================================================
# MongoDB Read Endpoints for Admin / Dashboard
# =============================================================================
@app.get("/transactions")
async def get_transactions(limit: int = 20):
    if mongo_tx_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    limit = max(1, min(limit, 200))
    cursor = mongo_tx_collection.find().sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    docs = [_serialize_doc(d) for d in docs]

    return {"count": len(docs), "items": docs}


@app.get("/reports")
async def get_reports(limit: int = 20):
    if mongo_report_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    limit = max(1, min(limit, 200))
    cursor = mongo_report_collection.find().sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)
    docs = [_serialize_doc(d) for d in docs]

    return {"count": len(docs), "items": docs}


@app.get("/transactions/{mongo_id}")
async def get_transaction_by_id(mongo_id: str):
    if mongo_tx_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    try:
        doc = await mongo_tx_collection.find_one({"_id": ObjectId(mongo_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid MongoDB ObjectId")

    if not doc:
        raise HTTPException(status_code=404, detail="Transaction not found")

    return _serialize_doc(doc)


@app.get("/reports/{mongo_id}")
async def get_report_by_id(mongo_id: str):
    if mongo_report_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    try:
        doc = await mongo_report_collection.find_one({"_id": ObjectId(mongo_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid MongoDB ObjectId")

    if not doc:
        raise HTTPException(status_code=404, detail="Report not found")

    return _serialize_doc(doc)