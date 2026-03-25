import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import tensorflow as tf

try:
    from core.settings import (
        AE_MODEL_PATH,
        ALT_AE,
        ALT_MLP,
        ALT_SCALER,
        BASE_DIR,
        CALIBRATOR_PATH,
        LOG_PATH,
        MLP_MODEL_PATH,
        SCALER_PATH,
        SHAP_BG_MAX,
        SHAP_BG_PATH,
        THRESHOLDS_PATH,
    )
    from core.state import state
    from utils.logger import get_logger
except ImportError:
    from backend.core.settings import (
        AE_MODEL_PATH,
        ALT_AE,
        ALT_MLP,
        ALT_SCALER,
        BASE_DIR,
        CALIBRATOR_PATH,
        LOG_PATH,
        MLP_MODEL_PATH,
        SCALER_PATH,
        SHAP_BG_MAX,
        SHAP_BG_PATH,
        THRESHOLDS_PATH,
    )
    from backend.core.state import state
    from backend.utils.logger import get_logger


logger = get_logger(__name__)


def first_existing(primary, alts) -> Optional[Any]:
    if primary.exists():
        return primary
    for path in alts:
        if path.exists():
            return path
    return None


def to_prob(pred: np.ndarray) -> float:
    pred = np.array(pred)

    if pred.ndim == 2 and pred.shape[1] == 2:
        return float(pred[0, 1])

    val = float(pred.reshape(-1)[0])
    if val < 0.0 or val > 1.0:
        return float(tf.sigmoid(val).numpy())
    return val


def ae_recon(ae: tf.keras.Model, x_scaled: np.ndarray) -> np.ndarray:
    return ae.predict(x_scaled, verbose=0)


def recon_error_mse(x_scaled: np.ndarray, x_hat: np.ndarray) -> np.ndarray:
    return np.mean(np.square(x_scaled - x_hat), axis=1)


def anomaly_score_fallback(re_err: float, thr_re: float) -> float:
    thr_re = float(max(thr_re, 1e-12))
    sigma = 0.20 * thr_re
    z = (re_err - thr_re) / max(sigma, 1e-12)
    return float(1.0 / (1.0 + np.exp(-z)))


def per_feature_sq_error(x_row: np.ndarray, x_hat_row: np.ndarray) -> np.ndarray:
    return np.square(x_row - x_hat_row).reshape(-1)


def feature_label(name: str) -> str:
    if name == "Time":
        return "transaction timing pattern"
    if name == "Amount":
        return "transaction amount pattern"
    if isinstance(name, str) and name.startswith("V"):
        return f"hidden transaction behaviour pattern ({name})"
    return str(name)


def risk_level(score_pct: float) -> str:
    if score_pct >= 70:
        return "HIGH"
    if score_pct >= 40:
        return "MEDIUM"
    return "LOW"


def join_labels(items: List[str]) -> str:
    cleaned = [str(x) for x in items if str(x).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def model_not_ready_message() -> str:
    return f"Model not ready: {state.last_init_error or 'unknown error'}"


def load_assets():
    state.last_init_error = None
    state.shap_explainer = None

    try:
        mlp_path = first_existing(MLP_MODEL_PATH, ALT_MLP)
        ae_path = first_existing(AE_MODEL_PATH, ALT_AE)

        if mlp_path is None:
            raise FileNotFoundError(f"MLP model not found. Tried: {MLP_MODEL_PATH} + {ALT_MLP}")
        if ae_path is None:
            raise FileNotFoundError(f"Autoencoder not found. Tried: {AE_MODEL_PATH} + {ALT_AE}")

        state.mlp_model = tf.keras.models.load_model(str(mlp_path), compile=False)
        state.ae_model = tf.keras.models.load_model(str(ae_path), compile=False)

        state.thresholds = {}
        if THRESHOLDS_PATH.exists():
            with open(THRESHOLDS_PATH, "r", encoding="utf-8") as file_obj:
                state.thresholds = json.load(file_obj) or {}

        feature_names = state.thresholds.get("features")
        if isinstance(feature_names, list) and len(feature_names) == 30:
            state.feature_names = feature_names
        else:
            state.feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

        scaler_path = first_existing(SCALER_PATH, ALT_SCALER)
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

            class LegacyScalerWrapper:
                def transform(self, x_2d: np.ndarray) -> np.ndarray:
                    x_2d = np.asarray(x_2d, dtype=np.float32)
                    time_part = time_scaler.transform(x_2d[:, [0]])
                    amount_part = amount_scaler.transform(x_2d[:, [-1]])
                    mid = x_2d[:, 1:29]
                    return np.concatenate([time_part, mid, amount_part], axis=1).astype(np.float32)

            state.scaler = LegacyScalerWrapper()
            shap_background_from_artifacts = artifacts.get("background", None)
        else:
            state.scaler = joblib.load(str(scaler_path))

        state.calibrator = joblib.load(str(CALIBRATOR_PATH)) if CALIBRATOR_PATH.exists() else None

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
                elif isinstance(shap_background_from_artifacts, np.ndarray):
                    bg = np.asarray(shap_background_from_artifacts, dtype=np.float32)
                    if bg.ndim != 2 or bg.shape[1] != 30:
                        raise ValueError(
                            f"Invalid preprocess background shape: {bg.shape} (expected (N,30))"
                        )
                    if bg.shape[0] > SHAP_BG_MAX:
                        bg = bg[:SHAP_BG_MAX]
                    bg_source = "preprocess_artifacts.pkl"
                else:
                    bg = np.zeros((min(50, SHAP_BG_MAX), 30), dtype=np.float32)

                try:
                    state.shap_explainer = shap.Explainer(state.mlp_model, bg)
                except Exception:
                    state.shap_explainer = shap.KernelExplainer(
                        lambda x: state.mlp_model.predict(np.array(x, dtype=np.float32), verbose=0),
                        bg,
                    )
                logger.info(f"SHAP initialized successfully using background source: {bg_source}")
            except Exception as shap_error:
                state.shap_explainer = None
                logger.warning(f"SHAP disabled: {shap_error}")

        state.last_init_error = None
        logger.info("Assets loaded successfully.")

    except Exception as exc:
        state.mlp_model = None
        state.ae_model = None
        state.scaler = None
        state.calibrator = None
        state.thresholds = {}
        state.shap_explainer = None
        state.last_init_error = str(exc)
        logger.exception(f"Asset loading failed: {exc}")


def xai_shap(
    x_scaled_row: np.ndarray,
    x_raw_row: np.ndarray,
    top_k: int,
    output_is_fraud: bool,
) -> List[Dict[str, Any]]:
    if state.shap_explainer is None:
        return []

    try:
        shap_values = state.shap_explainer(x_scaled_row)
        values = getattr(shap_values, "values", shap_values)
        values = np.array(values)

        if values.ndim == 3:
            class_idx = 1 if output_is_fraud and values.shape[2] > 1 else 0
            sv = values[0, :, class_idx]
        elif values.ndim == 2:
            sv = values[0]
        else:
            return []

        x_raw = x_raw_row[0]
        order = np.argsort(np.abs(sv))[::-1][:top_k]
        items = []
        for idx in order:
            items.append(
                {
                    "feature": state.feature_names[idx],
                    "feature_value": float(x_raw[idx]),
                    "impact": float(sv[idx]),
                    "direction": "raises risk" if float(sv[idx]) >= 0 else "lowers risk",
                }
            )
        return items
    except Exception as exc:
        logger.warning(f"SHAP generation failed: {exc}")
        return []


def xai_recon_pf(x_scaled_row: np.ndarray, x_hat_row: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
    errors = per_feature_sq_error(x_scaled_row, x_hat_row)
    order = np.argsort(errors)[::-1][:top_k]
    items = []
    for idx in order:
        items.append(
            {
                "feature": state.feature_names[idx],
                "error": float(errors[idx]),
            }
        )
    return items


def build_text_reason(
    shap_data: List[Dict[str, Any]],
    recon_data: List[Dict[str, Any]],
    time_value: float,
    amount_value: float,
    risk_score_pct: float,
) -> str:
    risk = risk_level(risk_score_pct)

    shap_labels = join_labels([feature_label(item["feature"]) for item in shap_data[:3]])
    recon_labels = join_labels([feature_label(item["feature"]) for item in recon_data[:3]])

    parts = [
        f"Transaction analysed with time feature value {time_value:.0f} and amount ${amount_value:.2f}.",
        f"Final risk level: {risk} ({risk_score_pct:.2f}%).",
    ]

    if shap_labels:
        parts.append(f"The classifier was most influenced by {shap_labels}.")
    if recon_labels:
        parts.append(f"The anomaly model saw unusual reconstruction error in {recon_labels}.")

    return " ".join(parts)


def _predict_scores(x_raw: np.ndarray) -> Tuple[np.ndarray, float, float, np.ndarray, float, float, float, float, float, bool]:
    x_scaled = state.scaler.transform(x_raw).astype(np.float32)

    try:
        input_name = state.mlp_model.input_names[0]
        raw_pred = state.mlp_model.predict({input_name: x_scaled}, verbose=0)
    except Exception:
        raw_pred = state.mlp_model.predict(x_scaled, verbose=0)

    p_raw = to_prob(raw_pred)
    output_is_fraud = os.getenv("MODEL_OUTPUT_IS_FRAUD", "1") == "1"
    p_fraud = p_raw if output_is_fraud else (1.0 - p_raw)

    if state.calibrator is not None:
        p_fraud = float(state.calibrator.transform([p_fraud])[0])
    else:
        p_fraud = float(p_fraud)

    x_hat = ae_recon(state.ae_model, x_scaled)
    re_err = float(recon_error_mse(x_scaled, x_hat)[0])

    thr_re = float(state.thresholds.get("ae_thr_val_norm_99.5pct", os.getenv("AE_THR", "0.01")))
    anomaly_score = anomaly_score_fallback(re_err, thr_re)

    weights = state.thresholds.get("weights", {"w_mlp": 0.6, "w_ae": 0.4})
    w_mlp = float(weights.get("w_mlp", 0.6))
    w_ae = float(weights.get("w_ae", 0.4))
    combined_score = float(w_mlp * p_fraud + w_ae * anomaly_score)

    combined_thr = state.thresholds.get("combined_thr_val_maxf1", {}).get("thr", None)
    combined_thr = (
        float(combined_thr)
        if combined_thr is not None
        else float(os.getenv("COMBINED_THR", "0.5"))
    )

    return x_scaled, p_raw, p_fraud, x_hat, re_err, thr_re, anomaly_score, combined_score, combined_thr, output_is_fraud


def build_prediction_response(tx, client_name: str) -> Dict[str, Any]:
    x_raw = np.array([[tx.Time] + tx.V_features + [tx.Amount]], dtype=np.float32)
    if x_raw.shape != (1, 30):
        raise ValueError("Input must be 30 features: Time + 28 V-features + Amount")

    (
        x_scaled,
        p_raw,
        p_fraud,
        x_hat,
        re_err,
        thr_re,
        anomaly_score,
        combined_score,
        combined_thr,
        output_is_fraud,
    ) = _predict_scores(x_raw)

    is_fraud = combined_score >= combined_thr
    weights = state.thresholds.get("weights", {"w_mlp": 0.6, "w_ae": 0.4})
    w_mlp = float(weights.get("w_mlp", 0.6))
    w_ae = float(weights.get("w_ae", 0.4))

    top_k = int(tx.top_k)
    shap_data = xai_shap(x_scaled, x_raw, top_k=top_k, output_is_fraud=output_is_fraud) if tx.include_xai else []
    recon_data = xai_recon_pf(x_scaled[0], x_hat[0], top_k) if tx.include_xai else []

    risk_score_pct = combined_score * 100.0
    explanation = (
        build_text_reason(
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
            f"{risk_level(risk_score_pct)} ({risk_score_pct:.2f}%)."
        )
    )

    return {
        "transaction_id": tx.transaction_id,
        "client_name": client_name,
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


def build_batch_prediction_response(req, client_name: str) -> Dict[str, Any]:
    txs = req.transactions
    do_xai = bool(req.include_xai)
    top_k = int(req.top_k)

    x_raw = np.array([[tx.Time] + tx.V_features + [tx.Amount] for tx in txs], dtype=np.float32)
    if x_raw.ndim != 2 or x_raw.shape[1] != 30:
        raise ValueError("Each transaction must contain 30 features: Time + 28 V + Amount")

    x_scaled = state.scaler.transform(x_raw).astype(np.float32)

    try:
        input_name = state.mlp_model.input_names[0]
        raw_pred = state.mlp_model.predict({input_name: x_scaled}, verbose=0)
    except Exception:
        raw_pred = state.mlp_model.predict(x_scaled, verbose=0)

    raw_pred = np.array(raw_pred)

    if raw_pred.ndim == 2 and raw_pred.shape[1] == 2:
        p_raw = raw_pred[:, 1].astype(np.float32)
    else:
        p_raw = raw_pred.reshape(-1).astype(np.float32)
        p_raw = np.where(
            (p_raw < 0.0) | (p_raw > 1.0),
            1.0 / (1.0 + np.exp(-p_raw)),
            p_raw,
        ).astype(np.float32)

    output_is_fraud = os.getenv("MODEL_OUTPUT_IS_FRAUD", "1") == "1"
    p_fraud = p_raw if output_is_fraud else (1.0 - p_raw)

    if state.calibrator is not None:
        p_fraud = state.calibrator.transform(p_fraud.tolist()).astype(np.float32)

    x_hat = state.ae_model.predict(x_scaled, verbose=0)
    re_err = np.mean(np.square(x_scaled - x_hat), axis=1).astype(np.float32)

    thr_re = float(state.thresholds.get("ae_thr_val_norm_99.5pct", os.getenv("AE_THR", "0.01")))
    thr_safe = max(thr_re, 1e-12)
    sigma = 0.20 * thr_safe
    z = (re_err - thr_safe) / max(sigma, 1e-12)
    anomaly_score = (1.0 / (1.0 + np.exp(-z))).astype(np.float32)

    weights = state.thresholds.get("weights", {"w_mlp": 0.6, "w_ae": 0.4})
    w_mlp = float(weights.get("w_mlp", 0.6))
    w_ae = float(weights.get("w_ae", 0.4))
    combined = (w_mlp * p_fraud + w_ae * anomaly_score).astype(np.float32)

    combined_thr = state.thresholds.get("combined_thr_val_maxf1", {}).get("thr", None)
    combined_thr = (
        float(combined_thr)
        if combined_thr is not None
        else float(os.getenv("COMBINED_THR", "0.5"))
    )

    is_fraud = combined >= combined_thr

    results: List[Dict[str, Any]] = []
    for index, tx in enumerate(txs):
        risk_score_pct = float(combined[index]) * 100.0

        item: Dict[str, Any] = {
            "index": index,
            "transaction_id": tx.transaction_id,
            "is_fraud": bool(is_fraud[index]),
            "status": "BLOCKED" if bool(is_fraud[index]) else "APPROVED",
            "risk_score": float(round(risk_score_pct, 2)),
            "fraud_prob": float(p_fraud[index]),
            "recon_error": float(re_err[index]),
            "anomaly_score": float(anomaly_score[index]),
            "combined_score": float(combined[index]),
        }

        if do_xai:
            shap_data = xai_shap(
                x_scaled[index:index + 1],
                x_raw[index:index + 1],
                top_k=top_k,
                output_is_fraud=output_is_fraud,
            )
            recon_data = xai_recon_pf(x_scaled[index], x_hat[index], top_k=top_k)
            item["xai_data"] = shap_data
            item["ae_xai_data"] = recon_data
            item["explanation"] = build_text_reason(
                shap_data,
                recon_data,
                time_value=float(tx.Time),
                amount_value=float(tx.Amount),
                risk_score_pct=risk_score_pct,
            )

        results.append(item)

    results.sort(key=lambda item: item["risk_score"], reverse=True)

    return {
        "client_name": client_name,
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


def build_report_response(req, client_name: str) -> Dict[str, Any]:
    x_raw = np.array([[req.Time] + req.V_features + [req.Amount]], dtype=np.float32)
    if x_raw.shape != (1, 30):
        raise ValueError("Input must be 30 features: Time + 28 V-features + Amount")

    (
        x_scaled,
        p_raw,
        p_fraud_cal,
        x_hat,
        re_err,
        thr_re,
        anomaly_score,
        combined_score,
        combined_thr,
        output_is_fraud,
    ) = _predict_scores(x_raw)

    review_margin = float(os.getenv("REVIEW_MARGIN", "0.05"))
    if combined_score >= combined_thr:
        decision = "BLOCKED"
    elif combined_score >= (combined_thr - review_margin):
        decision = "REVIEW"
    else:
        decision = "APPROVED"

    top_k = int(req.top_k)
    shap_data: List[Dict[str, Any]] = []
    if req.include_shap and state.shap_explainer is not None:
        shap_data = xai_shap(
            x_scaled,
            x_raw,
            top_k=top_k,
            output_is_fraud=output_is_fraud,
        )

    recon_data: List[Dict[str, Any]] = []
    if req.include_recon:
        recon_data = xai_recon_pf(x_scaled[0], x_hat[0], top_k)

    summary = build_text_reason(
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
        "client_name": client_name,
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
            "weights": {
                "w_mlp": float(state.thresholds.get("weights", {}).get("w_mlp", 0.6)),
                "w_ae": float(state.thresholds.get("weights", {}).get("w_ae", 0.4)),
            },
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
                "mlp_output_shape": getattr(state.mlp_model, "output_shape", None),
                "ae_output_shape": getattr(state.ae_model, "output_shape", None),
            },
            "xai": {
                "shap_available": state.shap_explainer is not None,
                "feature_names_len": len(state.feature_names),
                "shap_bg_path": str(SHAP_BG_PATH),
                "shap_bg_max": SHAP_BG_MAX,
            },
        },
    }

    if req.include_raw_features:
        card["input"] = {
            "Time": float(req.Time),
            "V_features": [float(value) for value in req.V_features],
            "Amount": float(req.Amount),
        }

    return card


def build_health_payload(mongo_config: Dict[str, Any], api_key_default_expiry_days: int, google_client_id: str, admin_setup_key: str) -> Dict[str, Any]:
    return {
        "mlp_loaded": state.mlp_model is not None,
        "ae_loaded": state.ae_model is not None,
        "scaler_loaded": state.scaler is not None,
        "calibrator_loaded": state.calibrator is not None,
        "shap_loaded": state.shap_explainer is not None,
        "mongodb": {
            "connected": state.mongo_client is not None and state.mongo_db is not None,
            **mongo_config,
        },
        "paths": {
            "base_dir": str(BASE_DIR),
            "mlp_model_path": str(first_existing(MLP_MODEL_PATH, ALT_MLP) or MLP_MODEL_PATH),
            "ae_model_path": str(first_existing(AE_MODEL_PATH, ALT_AE) or AE_MODEL_PATH),
            "scaler_path": str(first_existing(SCALER_PATH, ALT_SCALER) or SCALER_PATH),
            "calibrator_path": str(CALIBRATOR_PATH),
            "thresholds_path": str(THRESHOLDS_PATH),
            "log_path": str(LOG_PATH),
            "shap_bg_path": str(SHAP_BG_PATH),
        },
        "model_info": {
            "mlp_output_shape": getattr(state.mlp_model, "output_shape", None) if state.mlp_model else None,
            "mlp_input_names": getattr(state.mlp_model, "input_names", None) if state.mlp_model else None,
            "ae_output_shape": getattr(state.ae_model, "output_shape", None) if state.ae_model else None,
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
            "MONGO_DB_NAME": mongo_config["db_name"],
            "GOOGLE_CLIENT_ID_SET": bool(google_client_id),
            "ADMIN_SETUP_KEY_SET": bool(admin_setup_key and admin_setup_key != "change-me-in-env"),
            "API_KEY_DEFAULT_EXPIRY_DAYS": api_key_default_expiry_days,
        },
        "last_init_error": state.last_init_error,
    }
