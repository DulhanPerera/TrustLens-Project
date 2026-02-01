"""
TrustLens FastAPI API (MLP + Autoencoder + optional SHAP)

Put these files in the SAME folder as this app.py (backend/):
  - mlp_model.keras
  - autoencoder.keras
  - scaler.joblib
  - isotonic_calibrator.joblib   (optional)
  - thresholds.json              (optional)

Run:
  pip install -r requirements.txt
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import tensorflow as tf
import joblib

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------
# Optional: reduce TF log noise (set before TF does heavy work)
# ---------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------------------------------------------------------------------
# Paths (backend/ folder)
# ---------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

# Your actual filenames (from screenshot)
DEFAULT_MLP = BASE_DIR / "mlp_model.keras"
DEFAULT_AE = BASE_DIR / "autoencoder.keras"
DEFAULT_SCALER = BASE_DIR / "scaler.joblib"
DEFAULT_CALIBRATOR = BASE_DIR / "isotonic_calibrator.joblib"
DEFAULT_THRESHOLDS = BASE_DIR / "thresholds.json"

# Also allow env override
MLP_MODEL_PATH = Path(os.getenv("MLP_MODEL_PATH", str(DEFAULT_MLP)))
AE_MODEL_PATH = Path(os.getenv("AE_MODEL_PATH", str(DEFAULT_AE)))
SCALER_PATH = Path(os.getenv("SCALER_PATH", str(DEFAULT_SCALER)))
CALIBRATOR_PATH = Path(os.getenv("CALIBRATOR_PATH", str(DEFAULT_CALIBRATOR)))
THRESHOLDS_PATH = Path(os.getenv("THRESHOLDS_PATH", str(DEFAULT_THRESHOLDS)))

# Extra fallbacks (optional)
ALT_MLP = [BASE_DIR / "mlp_best.keras", BASE_DIR / "trustlens_model.keras"]
ALT_AE = [BASE_DIR / "ae_best.keras"]
ALT_SCALER = [BASE_DIR / "preprocess_artifacts.pkl"]  # legacy dict option

LOG_PATH = BASE_DIR / "trustlens_audit.log"

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# ---------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------
app = FastAPI(title="TrustLens API", description="Fraud Detection with MLP + Autoencoder + XAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # CRA
        "http://localhost:5173",  # Vite
        "http://localhost:5174",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Globals (loaded at startup)
# ---------------------------------------------------------------------
mlp_model: Optional[tf.keras.Model] = None
ae_model: Optional[tf.keras.Model] = None
scaler = None
calibrator = None
thresholds: Dict[str, Any] = {}

# Optional SHAP explainer (can be heavy)
shap_explainer = None
feature_names: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

last_init_error: Optional[str] = None


# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def _first_existing(primary: Path, alts: List[Path]) -> Optional[Path]:
    if primary.exists():
        return primary
    for p in alts:
        if p.exists():
            return p
    return None


def _to_prob(pred: np.ndarray) -> float:
    """
    Converts raw model output into probability in [0,1].
    Supports:
      - (1,2) softmax -> class 1 prob
      - (1,1) sigmoid prob
      - (1,1) logit -> sigmoid(logit)
    """
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
    """
    Smooth 0..1 anomaly score around threshold using a sigmoid.
    Not a true quantile; stable for demos when normal error distribution isn't stored.
    """
    thr_re = float(max(thr_re, 1e-12))
    sigma = 0.20 * thr_re
    z = (re_err - thr_re) / max(sigma, 1e-12)
    return float(1.0 / (1.0 + np.exp(-z)))


def _per_feature_sq_error(x_row: np.ndarray, x_hat_row: np.ndarray) -> np.ndarray:
    return np.square(x_row - x_hat_row).reshape(-1)


# ---------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------
class Transaction(BaseModel):
    Time: float
    V_features: List[float] = Field(..., min_items=28, max_items=28)
    Amount: float

    include_xai: bool = True
    top_k: int = Field(4, ge=1, le=15)


# ---------------------------------------------------------------------
# Startup: load assets
# ---------------------------------------------------------------------
@app.on_event("startup")
def load_assets():
    global mlp_model, ae_model, scaler, calibrator, thresholds
    global shap_explainer, feature_names, last_init_error

    last_init_error = None
    shap_explainer = None

    try:
        # Resolve model paths
        mlp_path = _first_existing(MLP_MODEL_PATH, ALT_MLP)
        ae_path = _first_existing(AE_MODEL_PATH, ALT_AE)

        if mlp_path is None:
            raise FileNotFoundError(f"MLP model not found. Tried: {MLP_MODEL_PATH} + {ALT_MLP}")
        if ae_path is None:
            raise FileNotFoundError(f"Autoencoder not found. Tried: {AE_MODEL_PATH} + {ALT_AE}")

        # ✅ IMPORTANT FIX:
        # compile=False avoids needing focal loss function during loading
        mlp_model = tf.keras.models.load_model(str(mlp_path), compile=False)
        ae_model = tf.keras.models.load_model(str(ae_path), compile=False)

        # thresholds.json (optional)
        thresholds = {}
        if THRESHOLDS_PATH.exists():
            with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                thresholds = json.load(f) or {}

        # feature names (optional)
        fn = thresholds.get("features")
        if isinstance(fn, list) and len(fn) == 30:
            feature_names = fn
        else:
            feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

        # Load scaler (required)
        scaler_path = _first_existing(SCALER_PATH, ALT_SCALER)
        if scaler_path is None:
            raise FileNotFoundError(f"Scaler not found. Tried: {SCALER_PATH} + {ALT_SCALER}")

        if scaler_path.name.endswith(".pkl"):
            # legacy dict option: {time_scaler, amount_scaler, ...}
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
            shap_background = artifacts.get("background", None)

        else:
            scaler = joblib.load(str(scaler_path))
            shap_background = None

        # Load isotonic calibrator (optional)
        calibrator = joblib.load(str(CALIBRATOR_PATH)) if CALIBRATOR_PATH.exists() else None

        # Optional SHAP init
        enable_shap = os.getenv("ENABLE_SHAP", "1") == "1"
        if enable_shap:
            try:
                import shap

                if shap_background is None:
                    bg = np.zeros((10, 30), dtype=np.float32)
                else:
                    bg = np.array(shap_background, dtype=np.float32)
                    if bg.ndim != 2 or bg.shape[1] != 30:
                        bg = np.zeros((10, 30), dtype=np.float32)

                shap_explainer = shap.DeepExplainer(mlp_model, bg)
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


# ---------------------------------------------------------------------
# XAI helpers
# ---------------------------------------------------------------------
def _xai_shap(x_scaled: np.ndarray, top_k: int) -> List[Dict[str, float]]:
    if shap_explainer is None:
        return []
    try:
        shap_values = shap_explainer.shap_values(x_scaled)
        if isinstance(shap_values, list):
            vals = np.abs(np.array(shap_values[0]).reshape(-1))
        else:
            vals = np.abs(np.array(shap_values).reshape(-1))

        fn = feature_names if len(feature_names) == len(vals) else (["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
        idx = np.argsort(vals)[-top_k:][::-1]
        return [{"name": fn[i], "impact": float(vals[i])} for i in idx]
    except Exception as e:
        logging.warning(f"SHAP compute failed: {e}")
        return []


def _xai_recon_pf(x_scaled_row: np.ndarray, x_hat_row: np.ndarray, top_k: int) -> List[Dict[str, float]]:
    errs = _per_feature_sq_error(x_scaled_row, x_hat_row)
    fn = feature_names if len(feature_names) == len(errs) else (["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
    idx = np.argsort(errs)[-top_k:][::-1]
    return [{"name": fn[i], "impact": float(errs[i])} for i in idx]


def _build_text_reason(shap_top: List[Dict[str, float]], recon_top: List[Dict[str, float]]) -> str:
    parts = []
    if shap_top:
        parts.append("MLP drivers: " + ", ".join([d["name"] for d in shap_top[:2]]))
    if recon_top:
        parts.append("Anomaly drivers: " + ", ".join([d["name"] for d in recon_top[:2]]))
    return " | ".join(parts) if parts else "Transaction scored using probability + anomaly score."


# ---------------------------------------------------------------------
# Predict endpoint
# ---------------------------------------------------------------------
@app.post("/predict")
async def predict(tx: Transaction, background_tasks: BackgroundTasks):
    if mlp_model is None or ae_model is None or scaler is None:
        raise HTTPException(status_code=503, detail=f"Model not ready: {last_init_error or 'unknown error'}")

    try:
        # Build raw input [Time, V1..V28, Amount]
        x_raw = np.array([[tx.Time] + tx.V_features + [tx.Amount]], dtype=np.float32)
        if x_raw.shape != (1, 30):
            raise ValueError("Input must be 30 features: Time + 28 V-features + Amount")

        # Scale
        x_scaled = scaler.transform(x_raw).astype(np.float32)

        # Predict MLP
        try:
            input_name = mlp_model.input_names[0]
            raw_pred = mlp_model.predict({input_name: x_scaled}, verbose=0)
        except Exception:
            raw_pred = mlp_model.predict(x_scaled, verbose=0)

        p_raw = _to_prob(raw_pred)

        # Interpret as P(fraud) by default
        output_is_fraud = os.getenv("MODEL_OUTPUT_IS_FRAUD", "1") == "1"
        p_fraud = p_raw if output_is_fraud else (1.0 - p_raw)

        # Calibrate (optional)
        if calibrator is not None:
            p_fraud = float(calibrator.transform([p_fraud])[0])
        else:
            p_fraud = float(p_fraud)

        # AE reconstruction error
        x_hat = _ae_recon(ae_model, x_scaled)
        re_err = float(_recon_error_mse(x_scaled, x_hat)[0])

        # AE threshold (from thresholds.json if exists)
        thr_re = float(thresholds.get("ae_thr_val_norm_99.5pct", os.getenv("AE_THR", "0.01")))
        anomaly_score = _anomaly_score_fallback(re_err, thr_re)

        # Combine signals
        w = thresholds.get("weights", {"w_mlp": 0.6, "w_ae": 0.4})
        w_mlp = float(w.get("w_mlp", 0.6))
        w_ae = float(w.get("w_ae", 0.4))

        combined_score = float(w_mlp * p_fraud + w_ae * anomaly_score)

        # Decision threshold (use thresholds.json if possible)
        combined_thr = thresholds.get("combined_thr_val_maxf1", {}).get("thr", None)
        combined_thr = float(combined_thr) if combined_thr is not None else float(os.getenv("COMBINED_THR", "0.5"))

        is_fraud = combined_score >= combined_thr

        # XAI
        top_k = int(tx.top_k)
        shap_data = _xai_shap(x_scaled, top_k) if tx.include_xai else []
        recon_data = _xai_recon_pf(x_scaled[0], x_hat[0], top_k) if tx.include_xai else []
        explanation = _build_text_reason(shap_data, recon_data) if tx.include_xai else "Scored using probability + anomaly."

        if is_fraud:
            background_tasks.add_task(
                logging.info,
                f"Fraud Alert: combined={combined_score:.6f}, p_fraud={p_fraud:.6f}, re={re_err:.6f}"
            )

        logging.info(
            f"predict: p_raw={p_raw:.6f} p_fraud={p_fraud:.6f} re={re_err:.6f} anom={anomaly_score:.6f} combined={combined_score:.6f} is_fraud={is_fraud}"
        )

        return {
            "is_fraud": bool(is_fraud),
            "status": "BLOCKED" if is_fraud else "APPROVED",
            "risk_score": round(combined_score * 100.0, 2),

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

    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal analysis failure: {str(e)}")


# ---------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "mlp_loaded": mlp_model is not None,
        "ae_loaded": ae_model is not None,
        "scaler_loaded": scaler is not None,
        "calibrator_loaded": calibrator is not None,
        "shap_loaded": shap_explainer is not None,
        "paths": {
            "base_dir": str(BASE_DIR),
            "mlp_model_path": str(_first_existing(MLP_MODEL_PATH, ALT_MLP) or MLP_MODEL_PATH),
            "ae_model_path": str(_first_existing(AE_MODEL_PATH, ALT_AE) or AE_MODEL_PATH),
            "scaler_path": str(_first_existing(SCALER_PATH, ALT_SCALER) or SCALER_PATH),
            "calibrator_path": str(CALIBRATOR_PATH),
            "thresholds_path": str(THRESHOLDS_PATH),
            "log_path": str(LOG_PATH),
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
        },
        "last_init_error": last_init_error,
    }
