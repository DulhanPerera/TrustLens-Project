"""
TrustLens FastAPI API (MLP + Autoencoder + optional SHAP)

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
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

# =============================================================================
# Standard Library Imports
# =============================================================================
import os           # Operating system interface for environment variables and file paths
import json         # JSON parsing for configuration files (thresholds.json)
import logging      # Logging framework for audit trail and debugging
from pathlib import Path                    # Object-oriented filesystem paths
from typing import List, Optional, Dict, Any  # Type hints for better code clarity
from datetime import datetime               # Timestamp generation for reports

# =============================================================================
# Third-Party Library Imports
# =============================================================================
import numpy as np      # Numerical computing for array operations and math
import tensorflow as tf # Deep learning framework for MLP and Autoencoder models
import joblib           # Serialization library for loading sklearn objects (scaler, calibrator)

# =============================================================================
# FastAPI Framework Imports
# =============================================================================
from fastapi import FastAPI, HTTPException, BackgroundTasks  # Web framework, error handling, async tasks
from fastapi.middleware.cors import CORSMiddleware           # Cross-Origin Resource Sharing for frontend access
from pydantic import BaseModel, Field                        # Data validation and request/response schemas

# ---------------------------------------------------------------------
# Optional: reduce TF log noise (set before TF does heavy work)
# ---------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# =============================================================================
# File Paths Configuration
# =============================================================================
# All model artifacts should be placed in the backend/ folder alongside this script.
# Paths can be overridden via environment variables for deployment flexibility.
# =============================================================================

# Base directory: resolves to the folder containing this app.py file
BASE_DIR = Path(__file__).resolve().parent

# Default file paths for required model artifacts
DEFAULT_MLP = BASE_DIR / "mlp_model.keras"           # MLP classifier for fraud probability
DEFAULT_AE = BASE_DIR / "autoencoder.keras"          # Autoencoder for anomaly detection
DEFAULT_SCALER = BASE_DIR / "scaler.joblib"          # Feature scaler (StandardScaler or similar)
DEFAULT_CALIBRATOR = BASE_DIR / "isotonic_calibrator.joblib"  # Probability calibrator (optional)
DEFAULT_THRESHOLDS = BASE_DIR / "thresholds.json"    # Decision thresholds and weights config

# Environment variable overrides allow custom paths in production/Docker deployments
MLP_MODEL_PATH = Path(os.getenv("MLP_MODEL_PATH", str(DEFAULT_MLP)))
AE_MODEL_PATH = Path(os.getenv("AE_MODEL_PATH", str(DEFAULT_AE)))
SCALER_PATH = Path(os.getenv("SCALER_PATH", str(DEFAULT_SCALER)))
CALIBRATOR_PATH = Path(os.getenv("CALIBRATOR_PATH", str(DEFAULT_CALIBRATOR)))
THRESHOLDS_PATH = Path(os.getenv("THRESHOLDS_PATH", str(DEFAULT_THRESHOLDS)))

# Alternative file paths for backward compatibility with different naming conventions
ALT_MLP = [BASE_DIR / "mlp_best.keras", BASE_DIR / "trustlens_model.keras"]
ALT_AE = [BASE_DIR / "ae_best.keras"]
ALT_SCALER = [BASE_DIR / "preprocess_artifacts.pkl"]  # Legacy format: dict with time_scaler/amount_scaler

# =============================================================================
# SHAP (Explainable AI) Background Data Configuration
# =============================================================================
# SHAP DeepExplainer requires a background dataset to compute feature attributions.
# Using real normal transaction samples (after scaling) produces more meaningful explanations.
# Fallback: zeros array if no background data is available (less interpretable).
# =============================================================================
SHAP_BG_PATH = Path(os.getenv("SHAP_BG_PATH", str(BASE_DIR / "shap_background.npy")))
SHAP_BG_MAX = int(os.getenv("SHAP_BG_MAX", "500"))  # Max samples to use (memory/speed tradeoff)

# Audit log file for tracking predictions and fraud alerts
LOG_PATH = BASE_DIR / "trustlens_audit.log"

# =============================================================================
# Logging Configuration
# =============================================================================
# All API events, predictions, and errors are logged to trustlens_audit.log.
# This provides an audit trail for compliance and debugging purposes.
# =============================================================================
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # Timestamp + level + message
)

# =============================================================================
# FastAPI Application Initialization
# =============================================================================
# TrustLens API provides fraud detection endpoints with explainable AI capabilities.
# The API combines MLP classification with Autoencoder anomaly detection.
# =============================================================================
app = FastAPI(
    title="TrustLens API",
    description="Fraud Detection with MLP + Autoencoder + XAI (Explainable AI)"
)

# =============================================================================
# CORS (Cross-Origin Resource Sharing) Middleware
# =============================================================================
# Enables frontend applications running on different ports/domains to access the API.
# Add production URLs here when deploying to ensure secure cross-origin requests.
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",           # React default development port
        "http://localhost:5173",           # Vite default development port
        "http://localhost:5174",           # Vite alternate port
        "https://trustlens-fyp.vercel.app",  # Production frontend URL
    ],
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers (Authorization, Content-Type, etc.)
)

# =============================================================================
# Global Variables (Loaded at Startup)
# =============================================================================
# These objects are loaded once when the API starts and reused for all requests.
# Using globals avoids reloading models for each request (significant performance gain).
# =============================================================================

# Machine Learning Models
mlp_model: Optional[tf.keras.Model] = None    # MLP classifier: predicts fraud probability
ae_model: Optional[tf.keras.Model] = None     # Autoencoder: detects anomalies via reconstruction error

# Preprocessing and Calibration
scaler = None                                  # Feature scaler: normalizes input features
calibrator = None                              # Isotonic calibrator: improves probability estimates (optional)
thresholds: Dict[str, Any] = {}                # Decision thresholds and ensemble weights from config

# Explainable AI
shap_explainer = None                          # SHAP DeepExplainer for feature attribution
# Feature names for the credit card fraud dataset: Time + 28 PCA components (V1-V28) + Amount
feature_names: List[str] = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Error tracking: stores the last initialization error message for health checks
last_init_error: Optional[str] = None


# =============================================================================
# Utility Functions
# =============================================================================
# Internal helper functions for model loading, prediction processing, and scoring.
# Prefixed with underscore to indicate they are private/internal to this module.
# =============================================================================


def _first_existing(primary: Path, alts: List[Path]) -> Optional[Path]:
    """
    Find the first existing file from a primary path and list of alternatives.
    
    This allows flexible file naming conventions while maintaining backward
    compatibility with different model file names.
    
    Args:
        primary: The preferred file path to check first
        alts: List of alternative paths to check if primary doesn't exist
        
    Returns:
        The first existing Path, or None if no files exist
    """
    if primary.exists():
        return primary
    for p in alts:
        if p.exists():
            return p
    return None


def _to_prob(pred: np.ndarray) -> float:
    """
    Convert model prediction output to a probability value [0, 1].
    
    Handles different model output formats:
    - Softmax output shape (1, 2): returns pred[0, 1] (fraud class probability)
    - Sigmoid output shape (1, 1): returns the value directly if in [0, 1]
    - Raw logits: applies sigmoid to convert to probability
    
    Args:
        pred: Model prediction array (various shapes supported)
        
    Returns:
        Float probability value between 0 and 1
    """
    pred = np.array(pred)

    # Softmax output: 2 classes [P(normal), P(fraud)] - take fraud probability
    if pred.ndim == 2 and pred.shape[1] == 2:
        return float(pred[0, 1])

    # Sigmoid output or raw logits
    val = float(pred.reshape(-1)[0])
    # If outside [0,1] range, treat as logits and apply sigmoid
    if val < 0.0 or val > 1.0:
        return float(tf.sigmoid(val).numpy())
    return val


def _ae_recon(ae: tf.keras.Model, x_scaled: np.ndarray) -> np.ndarray:
    """
    Generate autoencoder reconstruction of input features.
    
    The autoencoder attempts to reconstruct the input. Normal transactions
    should reconstruct well (low error), while anomalies will have higher
    reconstruction error.
    
    Args:
        ae: Trained autoencoder model
        x_scaled: Scaled input features, shape (batch_size, 30)
        
    Returns:
        Reconstructed features with same shape as input
    """
    return ae.predict(x_scaled, verbose=0)


def _recon_error_mse(x_scaled: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    """
    Calculate Mean Squared Error between original and reconstructed features.
    
    Higher MSE indicates the transaction deviates from normal patterns
    learned by the autoencoder (potential anomaly/fraud).
    
    Args:
        x_scaled: Original scaled features, shape (batch_size, 30)
        x_hat: Reconstructed features from autoencoder
        
    Returns:
        Array of MSE values, one per sample
    """
    return np.mean(np.square(x_scaled - x_hat), axis=1)


def _anomaly_score_fallback(re_err: float, thr_re: float) -> float:
    """
    Convert reconstruction error to anomaly score using sigmoid transformation.
    
    Uses a sigmoid function centered at the threshold to produce a smooth
    score in [0, 1]. Scores above 0.5 indicate the reconstruction error
    exceeds the normal threshold.
    
    Formula: sigmoid((re_err - threshold) / (0.2 * threshold))
    
    Args:
        re_err: Reconstruction error (MSE) for the sample
        thr_re: Threshold value (e.g., 99.5th percentile of normal transactions)
        
    Returns:
        Anomaly score between 0 (normal) and 1 (highly anomalous)
    """
    thr_re = float(max(thr_re, 1e-12))  # Prevent division by zero
    sigma = 0.20 * thr_re               # Width of sigmoid transition
    z = (re_err - thr_re) / max(sigma, 1e-12)  # Z-score relative to threshold
    return float(1.0 / (1.0 + np.exp(-z)))     # Sigmoid transformation


def _per_feature_sq_error(x_row: np.ndarray, x_hat_row: np.ndarray) -> np.ndarray:
    """
    Calculate squared error for each feature individually.
    
    Used for XAI to identify which features contributed most to
    the autoencoder's reconstruction error (anomaly detection).
    
    Args:
        x_row: Original scaled features for one sample (1D array)
        x_hat_row: Reconstructed features for one sample (1D array)
        
    Returns:
        Array of squared errors, one per feature (length 30)
    """
    return np.square(x_row - x_hat_row).reshape(-1)


# =============================================================================
# Pydantic Request Schemas
# =============================================================================
# These classes define the expected structure of API request bodies.
# Pydantic provides automatic validation and type conversion.
# =============================================================================


class Transaction(BaseModel):
    """
    Schema for a single transaction prediction request.
    
    The credit card fraud dataset has 30 features:
    - Time: Seconds elapsed since first transaction in dataset
    - V1-V28: 28 PCA-transformed features (anonymized for privacy)
    - Amount: Transaction amount in the original currency
    
    Attributes:
        Time: Transaction timestamp (seconds from dataset start)
        V_features: List of exactly 28 PCA-transformed features (V1-V28)
        Amount: Transaction amount
        include_xai: Whether to include SHAP/reconstruction explanations
        top_k: Number of top features to return in XAI output (1-15)
    """
    Time: float
    V_features: List[float] = Field(..., min_items=28, max_items=28)
    Amount: float

    include_xai: bool = True   # Enable explainable AI output
    top_k: int = Field(4, ge=1, le=15)  # Top K features for explanations


class BatchRequest(BaseModel):
    """
    Schema for batch prediction request (50-500 transactions).
    
    Batch processing is more efficient for analyzing multiple transactions
    at once. XAI is disabled by default due to computational overhead.
    
    Attributes:
        transactions: List of 50-500 Transaction objects
        include_xai: Whether to compute XAI for each transaction (slow)
        top_k: Number of top features for XAI output
    """
    transactions: List[Transaction] = Field(..., min_items=50, max_items=500)
    include_xai: bool = False  # Batch XAI is computationally expensive; default off
    top_k: int = Field(4, ge=1, le=15)


class ReportRequest(BaseModel):
    """
    Schema for detailed fraud analysis report (TrustLens card).
    
    Returns a comprehensive analysis including decision, confidence,
    all scoring signals, and detailed XAI explanations.
    
    Attributes:
        Time: Transaction timestamp
        V_features: 28 PCA features (V1-V28)
        Amount: Transaction amount
        top_k: Number of top features for explanations
        include_shap: Whether to include SHAP feature attributions
        include_recon: Whether to include reconstruction error analysis
        include_raw_features: Whether to echo input features in response
    """
    Time: float
    V_features: List[float] = Field(..., min_items=28, max_items=28)
    Amount: float

    top_k: int = Field(6, ge=1, le=15)  # More features for detailed report
    include_shap: bool = True           # SHAP explanations enabled by default
    include_recon: bool = True          # Reconstruction error analysis enabled
    include_raw_features: bool = False  # Optionally include original input in response


# =============================================================================
# Application Startup: Load Model Assets
# =============================================================================
# This function runs once when the FastAPI server starts.
# It loads all required models, scalers, and configuration into memory.
# Failures here are logged but don't crash the server (health endpoint reports status).
# =============================================================================


@app.on_event("startup")
def load_assets():
    """
    Load all ML model assets at application startup.
    
    Loads:
    1. MLP model: Neural network classifier for fraud probability
    2. Autoencoder: Anomaly detection via reconstruction error
    3. Scaler: Feature normalization (StandardScaler or legacy wrapper)
    4. Calibrator: Isotonic regression for probability calibration (optional)
    5. Thresholds: Decision boundaries and ensemble weights from JSON config
    6. SHAP explainer: Deep learning explainer for feature attribution
    
    On failure, models remain None and last_init_error stores the error message.
    The /health endpoint can be used to check initialization status.
    """
    global mlp_model, ae_model, scaler, calibrator, thresholds
    global shap_explainer, feature_names, last_init_error

    # Reset error state
    last_init_error = None
    shap_explainer = None

    try:
        # =====================================================================
        # Step 1: Locate and Load Neural Network Models
        # =====================================================================
        mlp_path = _first_existing(MLP_MODEL_PATH, ALT_MLP)
        ae_path = _first_existing(AE_MODEL_PATH, ALT_AE)

        if mlp_path is None:
            raise FileNotFoundError(f"MLP model not found. Tried: {MLP_MODEL_PATH} + {ALT_MLP}")
        if ae_path is None:
            raise FileNotFoundError(f"Autoencoder not found. Tried: {AE_MODEL_PATH} + {ALT_AE}")

        # Load models with compile=False to avoid requiring custom loss functions
        # (e.g., focal loss) at load time - we only need inference, not training
        mlp_model = tf.keras.models.load_model(str(mlp_path), compile=False)
        ae_model = tf.keras.models.load_model(str(ae_path), compile=False)

        # =====================================================================
        # Step 2: Load Decision Thresholds and Configuration
        # =====================================================================
        thresholds = {}
        if THRESHOLDS_PATH.exists():
            with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                thresholds = json.load(f) or {}

        # Load custom feature names if provided in thresholds.json
        fn = thresholds.get("features")
        if isinstance(fn, list) and len(fn) == 30:
            feature_names = fn
        else:
            # Default feature names for credit card fraud dataset
            feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

        # =====================================================================
        # Step 3: Load Feature Scaler
        # =====================================================================
        scaler_path = _first_existing(SCALER_PATH, ALT_SCALER)
        if scaler_path is None:
            raise FileNotFoundError(f"Scaler not found. Tried: {SCALER_PATH} + {ALT_SCALER}")

        shap_background_from_artifacts = None  # May be populated from legacy artifacts

        # Handle two scaler formats:
        # 1. Legacy .pkl: dict with separate time_scaler and amount_scaler
        # 2. Modern .joblib: single StandardScaler for all 30 features
        if scaler_path.name.endswith(".pkl"):
            # Legacy format: separate scalers for Time and Amount columns
            artifacts = joblib.load(str(scaler_path))
            if not isinstance(artifacts, dict):
                raise ValueError("preprocess_artifacts.pkl must be a dict.")

            time_scaler = artifacts.get("time_scaler")
            amount_scaler = artifacts.get("amount_scaler")
            if time_scaler is None or amount_scaler is None:
                raise ValueError("preprocess_artifacts.pkl missing time_scaler/amount_scaler")

            class _LegacyScalerWrapper:
                """
                Wrapper to provide unified transform() interface for legacy scalers.
                Applies time_scaler to column 0, amount_scaler to column 29,
                and passes V1-V28 (columns 1-28) through unchanged.
                """
                def transform(self, X_2d: np.ndarray) -> np.ndarray:
                    X_2d = np.asarray(X_2d, dtype=np.float32)
                    t = time_scaler.transform(X_2d[:, [0]])    # Scale Time column
                    a = amount_scaler.transform(X_2d[:, [-1]]) # Scale Amount column
                    mid = X_2d[:, 1:29]                        # V1-V28 pass-through
                    return np.concatenate([t, mid, a], axis=1).astype(np.float32)

            scaler = _LegacyScalerWrapper()
            shap_background_from_artifacts = artifacts.get("background", None)
        else:
            # Modern format: single scaler for all features
            scaler = joblib.load(str(scaler_path))

        # =====================================================================
        # Step 4: Load Probability Calibrator (Optional)
        # =====================================================================
        # Isotonic calibration improves probability estimates from neural networks
        calibrator = joblib.load(str(CALIBRATOR_PATH)) if CALIBRATOR_PATH.exists() else None

        # -----------------------------
        # ✅ Improved SHAP initialization
        # -----------------------------
        enable_shap = os.getenv("ENABLE_SHAP", "1") == "1"
        if enable_shap:
            try:
                import shap  # local import so API runs even without shap installed

                bg_source = "zeros"

                # 1) Prefer shap_background.npy (recommended: real NORMAL samples AFTER scaling)
                if SHAP_BG_PATH.exists():
                    bg = np.load(str(SHAP_BG_PATH)).astype(np.float32)
                    if bg.ndim != 2 or bg.shape[1] != 30:
                        raise ValueError(f"Invalid shap_background.npy shape: {bg.shape} (expected (N,30))")
                    if bg.shape[0] > SHAP_BG_MAX:
                        bg = bg[:SHAP_BG_MAX]
                    bg_source = f"file:{SHAP_BG_PATH.name}"

                # 2) Else use legacy artifacts background if present (assumed already scaled)
                elif shap_background_from_artifacts is not None:
                    bg = np.array(shap_background_from_artifacts, dtype=np.float32)
                    if bg.ndim != 2 or bg.shape[1] != 30:
                        bg = np.zeros((10, 30), dtype=np.float32)
                        bg_source = "zeros(fallback_bad_artifacts_bg)"
                    else:
                        if bg.shape[0] > SHAP_BG_MAX:
                            bg = bg[:SHAP_BG_MAX]
                        bg_source = "artifacts:background"

                # 3) Else fallback (least meaningful)
                else:
                    bg = np.zeros((10, 30), dtype=np.float32)

                shap_explainer = shap.DeepExplainer(mlp_model, bg)
                logging.info(f"SHAP initialized. background_source={bg_source} shape={tuple(bg.shape)}")

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
# XAI helpers (Improved SHAP)
# ---------------------------------------------------------------------
def _xai_shap(
    x_scaled: np.ndarray,
    x_raw: np.ndarray,
    top_k: int,
    output_is_fraud: bool,
) -> List[Dict[str, Any]]:
    """
    More meaningful SHAP output:
      - uses signed SHAP values (direction: increase/decrease fraud risk)
      - includes raw + scaled feature values
      - ensures '+' means increases fraud risk (if model outputs P(normal), sign is flipped)

    Note: On this dataset V1..V28 are PCA components, so they won't map to real-world semantics,
    but the "increase/decrease" and values still make the explanation clearer.
    """
    if shap_explainer is None:
        return []

    try:
        shap_values = shap_explainer.shap_values(x_scaled)

        # signed SHAP values
        if isinstance(shap_values, list):
            vals = np.array(shap_values[0]).reshape(-1)
        else:
            vals = np.array(shap_values).reshape(-1)

        # Ensure "+" means "fraud risk increases"
        if not output_is_fraud:
            vals = -vals

        abs_vals = np.abs(vals)

        fn = feature_names if len(feature_names) == len(vals) else (["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
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
    """
    Identify top-K features with highest reconstruction error (anomaly drivers).
    
    Features with high reconstruction error indicate values that deviate
    from normal patterns learned by the autoencoder.
    
    Args:
        x_scaled_row: Original scaled features for one sample
        x_hat_row: Autoencoder reconstruction for one sample
        top_k: Number of top features to return
        
    Returns:
        List of dicts with 'name' and 'impact' (squared error) for top-K features
    """
    errs = _per_feature_sq_error(x_scaled_row, x_hat_row)
    fn = feature_names if len(feature_names) == len(errs) else (["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"])
    # Sort by error descending and take top K
    idx = np.argsort(errs)[-top_k:][::-1]
    return [{"name": fn[i], "impact": float(errs[i])} for i in idx]


def _build_text_reason(
    shap_top: List[Dict[str, Any]],
    recon_top: List[Dict[str, Any]],
    time_value: Optional[float] = None,
    amount_value: Optional[float] = None,
) -> str:
    """
    Text explanation that:
      - ALWAYS mentions Time + Amount
      - Summarizes SHAP direction: fraud risk ↑ / ↓
      - Adds anomaly drivers from reconstruction error
    """
    parts: List[str] = []

    # Always show Time/Amount context
    ctx: List[str] = []
    if time_value is not None:
        ctx.append(f"Time={float(time_value):.0f}")
    if amount_value is not None:
        ctx.append(f"Amount={float(amount_value):.2f}")
    if ctx:
        parts.append(", ".join(ctx))

    # Directional SHAP summary
    if shap_top:
        inc = [d["name"] for d in shap_top if float(d.get("signed", 0.0)) > 0][:2]
        dec = [d["name"] for d in shap_top if float(d.get("signed", 0.0)) < 0][:2]

        if inc:
            parts.append("Fraud risk ↑ mainly due to " + ", ".join(inc))
        if dec:
            parts.append("Fraud risk ↓ mainly due to " + ", ".join(dec))

    # AE anomaly drivers
    if recon_top:
        parts.append("Anomaly drivers: " + ", ".join([d["name"] for d in recon_top[:2]]))

    return " | ".join(parts) if parts else "Transaction scored using probability + anomaly score."


# =============================================================================
# API Endpoint: Single Transaction Prediction
# =============================================================================
# POST /predict - Analyze a single transaction for fraud
# Returns: fraud decision, risk score, and optional XAI explanations
# =============================================================================


@app.post("/predict")
async def predict(tx: Transaction, background_tasks: BackgroundTasks):
    """
    Predict fraud probability for a single transaction.
    
    Processing Pipeline:
    1. Validate and scale input features
    2. MLP prediction: fraud probability
    3. Autoencoder: reconstruction error → anomaly score
    4. Ensemble: combine MLP and AE scores with weighted average
    5. Decision: compare combined score against threshold
    6. XAI (optional): SHAP attributions + reconstruction error analysis
    
    Args:
        tx: Transaction data with 30 features and XAI options
        background_tasks: FastAPI background task manager for async logging
        
    Returns:
        JSON with is_fraud, status, risk_score, scores, thresholds, and XAI data
    """
    # Check if models are loaded and ready
    if mlp_model is None or ae_model is None or scaler is None:
        raise HTTPException(status_code=503, detail=f"Model not ready: {last_init_error or 'unknown error'}")

    try:
        # =====================================================================
        # Step 1: Prepare and Validate Input Features
        # =====================================================================
        # Combine all features into a single array: [Time, V1-V28, Amount]
        x_raw = np.array([[tx.Time] + tx.V_features + [tx.Amount]], dtype=np.float32)
        if x_raw.shape != (1, 30):
            raise ValueError("Input must be 30 features: Time + 28 V-features + Amount")

        # Apply feature scaling (normalization)
        x_scaled = scaler.transform(x_raw).astype(np.float32)

        # =====================================================================
        # Step 2: MLP Fraud Probability Prediction
        # =====================================================================
        # Try dict-style input first (for models with named inputs), fallback to array
        try:
            input_name = mlp_model.input_names[0]
            raw_pred = mlp_model.predict({input_name: x_scaled}, verbose=0)
        except Exception:
            raw_pred = mlp_model.predict(x_scaled, verbose=0)

        # Convert raw prediction to probability
        p_raw = _to_prob(raw_pred)

        # Handle model output interpretation:
        # If MODEL_OUTPUT_IS_FRAUD=1, output is P(fraud)
        # If MODEL_OUTPUT_IS_FRAUD=0, output is P(normal), so we flip it
        output_is_fraud = os.getenv("MODEL_OUTPUT_IS_FRAUD", "1") == "1"
        p_fraud = p_raw if output_is_fraud else (1.0 - p_raw)

        # Apply probability calibration if calibrator is available
        # Isotonic calibration improves reliability of probability estimates
        if calibrator is not None:
            p_fraud = float(calibrator.transform([p_fraud])[0])
        else:
            p_fraud = float(p_fraud)

        # =====================================================================
        # Step 3: Autoencoder Anomaly Detection
        # =====================================================================
        # Generate reconstruction and calculate error
        x_hat = _ae_recon(ae_model, x_scaled)
        re_err = float(_recon_error_mse(x_scaled, x_hat)[0])

        # Get anomaly threshold (99.5th percentile of normal transaction errors)
        thr_re = float(thresholds.get("ae_thr_val_norm_99.5pct", os.getenv("AE_THR", "0.01")))
        # Convert reconstruction error to anomaly score [0, 1]
        anomaly_score = _anomaly_score_fallback(re_err, thr_re)

        # =====================================================================
        # Step 4: Ensemble Score Combination
        # =====================================================================
        # Weighted average of MLP probability and AE anomaly score
        # Default: 60% MLP + 40% AE (configurable via thresholds.json)
        w = thresholds.get("weights", {"w_mlp": 0.6, "w_ae": 0.4})
        w_mlp = float(w.get("w_mlp", 0.6))
        w_ae = float(w.get("w_ae", 0.4))
        combined_score = float(w_mlp * p_fraud + w_ae * anomaly_score)

        # =====================================================================
        # Step 5: Fraud Decision
        # =====================================================================
        # Get decision threshold (optimized for max F1 score on validation set)
        combined_thr = thresholds.get("combined_thr_val_maxf1", {}).get("thr", None)
        combined_thr = float(combined_thr) if combined_thr is not None else float(os.getenv("COMBINED_THR", "0.5"))

        # Make binary fraud decision
        is_fraud = combined_score >= combined_thr

        # XAI (single)
        top_k = int(tx.top_k)
        shap_data = (
            _xai_shap(x_scaled, x_raw, top_k=top_k, output_is_fraud=output_is_fraud)
            if tx.include_xai else []
        )
        recon_data = _xai_recon_pf(x_scaled[0], x_hat[0], top_k) if tx.include_xai else []
        explanation = (
            _build_text_reason(shap_data, recon_data, time_value=tx.Time, amount_value=tx.Amount)
            if tx.include_xai else f"Time={tx.Time:.0f}, Amount={tx.Amount:.2f} | Scored using probability + anomaly."
        )

        if is_fraud:
            background_tasks.add_task(
                logging.info,
                f"Fraud Alert: combined={combined_score:.6f}, p_fraud={p_fraud:.6f}, re={re_err:.6f}",
            )

        logging.info(
            f"predict: p_raw={p_raw:.6f} p_fraud={p_fraud:.6f} re={re_err:.6f} "
            f"anom={anomaly_score:.6f} combined={combined_score:.6f} is_fraud={is_fraud}"
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
            "xai_data": shap_data,         # now includes signed + direction + values
            "ae_xai_data": recon_data,
        }

    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal analysis failure: {str(e)}")


# =============================================================================
# API Endpoint: Batch Transaction Prediction
# =============================================================================
# POST /predict_batch - Analyze 50-500 transactions in a single request
# Returns: results sorted by risk_score (highest risk first)
# Note: XAI is disabled by default for performance (enable with include_xai=true)
# =============================================================================


@app.post("/predict_batch")
async def predict_batch(req: BatchRequest):
    """
    Predict fraud probability for multiple transactions (batch processing).
    
    Optimized for throughput with vectorized operations. Results are sorted
    by risk_score in descending order to prioritize high-risk transactions.
    
    Args:
        req: BatchRequest with 50-500 transactions and XAI options
        
    Returns:
        JSON with count, sorted results, and threshold configuration
    """
    if mlp_model is None or ae_model is None or scaler is None:
        raise HTTPException(status_code=503, detail=f"Model not ready: {last_init_error or 'unknown error'}")

    try:
        txs = req.transactions
        do_xai = bool(req.include_xai)  # XAI is computationally expensive
        top_k = int(req.top_k)

        # =====================================================================
        # Step 1: Prepare Batch Input (Vectorized)
        # =====================================================================
        # Convert all transactions to a 2D array for efficient batch processing
        X_raw = np.array([[t.Time] + t.V_features + [t.Amount] for t in txs], dtype=np.float32)
        if X_raw.ndim != 2 or X_raw.shape[1] != 30:
            raise ValueError("Each transaction must contain 30 features: Time + 28 V + Amount")

        # Scale all features at once
        X_scaled = scaler.transform(X_raw).astype(np.float32)

        # =====================================================================
        # Step 2: Vectorized MLP Prediction
        # =====================================================================
        try:
            input_name = mlp_model.input_names[0]
            raw_pred = mlp_model.predict({input_name: X_scaled}, verbose=0)
        except Exception:
            raw_pred = mlp_model.predict(X_scaled, verbose=0)

        raw_pred = np.array(raw_pred)

        # =====================================================================
        # Step 3: Convert Predictions to Probabilities (Batch)
        # =====================================================================
        # Handle softmax (2-class) vs sigmoid (1-class) output
        if raw_pred.ndim == 2 and raw_pred.shape[1] == 2:
            # Softmax: take fraud class probability
            p_raw = raw_pred[:, 1].astype(np.float32)
        else:
            # Sigmoid or logits: apply sigmoid if values outside [0,1]
            p_raw = raw_pred.reshape(-1).astype(np.float32)
            p_raw = np.where(
                (p_raw < 0.0) | (p_raw > 1.0),
                1.0 / (1.0 + np.exp(-p_raw)),  # Sigmoid for logits
                p_raw
            ).astype(np.float32)

        # Flip probability if model outputs P(normal) instead of P(fraud)
        output_is_fraud = os.getenv("MODEL_OUTPUT_IS_FRAUD", "1") == "1"
        p_fraud = p_raw if output_is_fraud else (1.0 - p_raw)

        # Apply calibration to all probabilities at once
        if calibrator is not None:
            p_fraud = calibrator.transform(p_fraud.tolist()).astype(np.float32)

        # =====================================================================
        # Step 4: Vectorized Autoencoder Anomaly Detection
        # =====================================================================
        X_hat = ae_model.predict(X_scaled, verbose=0)
        # Calculate reconstruction error for all samples at once
        re_err = np.mean(np.square(X_scaled - X_hat), axis=1).astype(np.float32)

        # Convert reconstruction errors to anomaly scores using sigmoid transformation
        thr_re = float(thresholds.get("ae_thr_val_norm_99.5pct", os.getenv("AE_THR", "0.01")))
        thr_safe = max(thr_re, 1e-12)     # Prevent division by zero
        sigma = 0.20 * thr_safe            # Width of sigmoid transition
        z = (re_err - thr_safe) / max(sigma, 1e-12)  # Z-scores for all samples
        anomaly_score = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)  # Vectorized sigmoid

        # =====================================================================
        # Step 5: Ensemble Score Combination (Batch)
        # =====================================================================
        w = thresholds.get("weights", {"w_mlp": 0.6, "w_ae": 0.4})
        w_mlp = float(w.get("w_mlp", 0.6))
        w_ae = float(w.get("w_ae", 0.4))
        combined = (w_mlp * p_fraud + w_ae * anomaly_score).astype(np.float32)

        # Get decision threshold
        combined_thr = thresholds.get("combined_thr_val_maxf1", {}).get("thr", None)
        combined_thr = float(combined_thr) if combined_thr is not None else float(os.getenv("COMBINED_THR", "0.5"))

        # Vectorized fraud decision for all samples
        is_fraud = combined >= combined_thr

        # =====================================================================
        # Step 6: Build Results with Optional XAI
        # =====================================================================
        results: List[Dict[str, Any]] = []
        for i in range(len(txs)):
            item: Dict[str, Any] = {
                "index": i,
                "is_fraud": bool(is_fraud[i]),
                "status": "BLOCKED" if bool(is_fraud[i]) else "APPROVED",
                "risk_score": float(round(float(combined[i]) * 100.0, 2)),
                "fraud_prob": float(p_fraud[i]),
                "recon_error": float(re_err[i]),
                "anomaly_score": float(anomaly_score[i]),
                "combined_score": float(combined[i]),
            }

            # Optional per-row explanations (slow)
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
                )

            results.append(item)

        # Rank by risk_score desc
        results.sort(key=lambda x: x["risk_score"], reverse=True)

        return {
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

    except Exception as e:
        logging.error(f"Batch Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# =============================================================================
# API Endpoint: Detailed Fraud Analysis Report (TrustLens Card)
# =============================================================================
# POST /report - Generate comprehensive fraud analysis with explanations
# Returns: decision (APPROVED/REVIEW/BLOCKED), confidence, all scores, and XAI data
# This is the primary endpoint for the TrustLens frontend dashboard
# =============================================================================


@app.post("/report")
async def report(req: ReportRequest):
    """
    Generate a detailed fraud analysis report (TrustLens Card).
    
    Provides a comprehensive analysis including:
    - 3-level decision: APPROVED, REVIEW (borderline), or BLOCKED
    - Confidence score based on distance from threshold
    - All intermediate signals (MLP prob, anomaly score, etc.)
    - SHAP feature attributions with direction (increase/decrease fraud risk)
    - Reconstruction error analysis (anomaly drivers)
    - Human-readable text summary
    - Metadata (timestamps, model info, XAI configuration)
    
    Args:
        req: ReportRequest with transaction features and XAI options
        
    Returns:
        JSON TrustLens card with complete analysis and explanations
    """
    if mlp_model is None or ae_model is None or scaler is None:
        raise HTTPException(status_code=503, detail=f"Model not ready: {last_init_error or 'unknown error'}")

    try:
        # Prepare and validate input
        x_raw = np.array([[req.Time] + req.V_features + [req.Amount]], dtype=np.float32)
        if x_raw.shape != (1, 30):
            raise ValueError("Input must be 30 features: Time + 28 V-features + Amount")

        x_scaled = scaler.transform(x_raw).astype(np.float32)

        # MLP
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

        # AE
        x_hat = _ae_recon(ae_model, x_scaled)
        re_err = float(_recon_error_mse(x_scaled, x_hat)[0])

        thr_re = float(thresholds.get("ae_thr_val_norm_99.5pct", os.getenv("AE_THR", "0.01")))
        anomaly_score = _anomaly_score_fallback(re_err, thr_re)

        # Combine
        w = thresholds.get("weights", {"w_mlp": 0.6, "w_ae": 0.4})
        w_mlp = float(w.get("w_mlp", 0.6))
        w_ae = float(w.get("w_ae", 0.4))
        combined_score = float(w_mlp * p_fraud_cal + w_ae * anomaly_score)

        combined_thr = thresholds.get("combined_thr_val_maxf1", {}).get("thr", None)
        combined_thr = float(combined_thr) if combined_thr is not None else float(os.getenv("COMBINED_THR", "0.5"))

        # =====================================================================
        # 3-Level Decision Logic
        # =====================================================================
        # BLOCKED: combined_score >= threshold (high fraud risk)
        # REVIEW: combined_score in [threshold - margin, threshold) (borderline)
        # APPROVED: combined_score < threshold - margin (low fraud risk)
        # The review margin creates a "gray zone" for human analyst review
        # =====================================================================
        review_margin = float(os.getenv("REVIEW_MARGIN", "0.05"))  # Default 5% margin
        if combined_score >= combined_thr:
            decision = "BLOCKED"    # High risk - block transaction
        elif combined_score >= (combined_thr - review_margin):
            decision = "REVIEW"     # Borderline - flag for manual review
        else:
            decision = "APPROVED"   # Low risk - approve transaction

        top_k = int(req.top_k)

        # XAI
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
            recon_data = _xai_recon_pf(x_scaled[0], x_hat[0], top_k=top_k)

        summary = _build_text_reason(
            shap_data,
            recon_data,
            time_value=req.Time,
            amount_value=req.Amount,
        )

        # Confidence heuristic: distance from threshold (0..1)
        dist = abs(combined_score - combined_thr)
        confidence = float(min(1.0, dist / max(combined_thr, 1e-6)))

        card: Dict[str, Any] = {
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
                "shap_top": shap_data,       # signed + direction + values
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

        return card

    except Exception as e:
        logging.error(f"Report Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


# =============================================================================
# API Endpoint: Health Check
# =============================================================================
# GET /health - Check API status and model loading state
# Used for monitoring, debugging, and deployment health checks
# =============================================================================


@app.get("/health")
def health():
    """
    Health check endpoint for monitoring and debugging.
    
    Returns comprehensive status information including:
    - Model loading status (MLP, AE, scaler, calibrator, SHAP)
    - File paths for all artifacts
    - Model architecture info (shapes, input names)
    - Environment variable configuration
    - Last initialization error (if any)
    
    Use this endpoint to:
    - Verify API is running and models are loaded
    - Debug configuration issues
    - Monitor deployment health (Kubernetes, Docker, etc.)
    
    Returns:
        JSON with complete health status and configuration details
    """
    return {
        # Model loading status flags
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
        },
        "last_init_error": last_init_error,
    }
