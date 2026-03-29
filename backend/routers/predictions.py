"""
Prediction routes live here.
They run single checks, batch checks, and report generation.
"""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request

try:
    from core.state import state
    from schemas.request_models import BatchRequest, ReportRequest, Transaction
    from services.auth_service import verify_api_key
    from services.mongo_service import (
        log_activity,
        log_request_event,
        save_batch_to_mongo,
        save_prediction_to_mongo,
        save_report_to_mongo,
    )
    from services.prediction_service import (
        build_batch_prediction_response,
        build_prediction_response,
        build_report_response,
        model_not_ready_message,
    )
    from utils.logger import get_logger
except ImportError:
    from backend.core.state import state
    from backend.schemas.request_models import BatchRequest, ReportRequest, Transaction
    from backend.services.auth_service import verify_api_key
    from backend.services.mongo_service import (
        log_activity,
        log_request_event,
        save_batch_to_mongo,
        save_prediction_to_mongo,
        save_report_to_mongo,
    )
    from backend.services.prediction_service import (
        build_batch_prediction_response,
        build_prediction_response,
        build_report_response,
        model_not_ready_message,
    )
    from backend.utils.logger import get_logger


router = APIRouter()
logger = get_logger(__name__)


def _ensure_models_ready():
    # Stop requests early if the models did not load on startup.
    if state.mlp_model is None or state.ae_model is None or state.scaler is None:
        raise HTTPException(status_code=503, detail=model_not_ready_message())


@router.post("/predict")
async def predict(
    tx: Transaction,
    background_tasks: BackgroundTasks,
    request: Request,
    client=Depends(verify_api_key),
):
    # Run the main fraud check for one transaction.
    _ensure_models_ready()

    try:
        response = build_prediction_response(tx, client_name=client["client_name"])

        if response["is_fraud"]:
            background_tasks.add_task(
                logger.info,
                (
                    "Fraud Alert: "
                    f"combined={response['combined_score']:.6f}, "
                    f"p_fraud={response['fraud_prob']:.6f}, "
                    f"re={response['recon_error']:.6f}"
                ),
            )

        logger.info(
            f"predict: client={client['client_name']} tx_id={tx.transaction_id} "
            f"p_raw={response['mlp_prob_raw']:.6f} p_fraud={response['fraud_prob']:.6f} "
            f"re={response['recon_error']:.6f} anom={response['anomaly_score']:.6f} "
            f"combined={response['combined_score']:.6f} is_fraud={response['is_fraud']}"
        )

        mongo_id = await save_prediction_to_mongo(tx, response, client_name=client["client_name"])
        response["mongo_id"] = mongo_id

        await log_activity(
            action="transaction_predicted",
            actor=client["client_name"],
            details={
                "transaction_id": tx.transaction_id,
                "mongo_id": mongo_id,
                "status": response["status"],
                "risk_score": response["risk_score"],
            },
        )

        await log_request_event(
            client_name=client["client_name"],
            endpoint=str(request.url.path),
            method=request.method,
            status="success",
            details={
                "transaction_id": tx.transaction_id,
                "mongo_id": mongo_id,
                "decision": response["status"],
            },
        )

        return response

    except Exception as exc:
        await log_request_event(
            client_name=client["client_name"],
            endpoint=str(request.url.path),
            method=request.method,
            status="error",
            details={"error": str(exc), "transaction_id": tx.transaction_id},
        )
        logger.error(f"Prediction Error: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"Internal analysis failure: {str(exc)}")


@router.post("/predict_batch")
async def predict_batch(
    req: BatchRequest,
    request: Request,
    client=Depends(verify_api_key),
):
    # Run many transactions in one request for external systems.
    _ensure_models_ready()

    try:
        batch_response = build_batch_prediction_response(req, client_name=client["client_name"])
        fraud_count = sum(1 for item in batch_response["results"] if item.get("is_fraud") is True)
        approved_count = sum(1 for item in batch_response["results"] if item.get("is_fraud") is False)

        logger.info(
            f"predict_batch: client={client['client_name']} count={len(batch_response['results'])} "
            f"blocked={fraud_count} approved={approved_count}"
        )

        for item in batch_response["results"]:
            logger.info(
                f"predict_batch_item: client={client['client_name']} "
                f"tx_id={item.get('transaction_id')} "
                f"p_fraud={float(item.get('fraud_prob', 0.0)):.6f} "
                f"re={float(item.get('recon_error', 0.0)):.6f} "
                f"anom={float(item.get('anomaly_score', 0.0)):.6f} "
                f"combined={float(item.get('combined_score', 0.0)):.6f} "
                f"is_fraud={item.get('is_fraud')}"
            )

        mongo_id = await save_batch_to_mongo(req, batch_response, client_name=client["client_name"])
        batch_response["mongo_id"] = mongo_id

        await log_activity(
            action="batch_prediction",
            actor=client["client_name"],
            details={
                "count": len(batch_response["results"]),
                "fraud_count": fraud_count,
                "approved_count": approved_count,
                "mongo_id": mongo_id,
            },
        )

        await log_request_event(
            client_name=client["client_name"],
            endpoint=str(request.url.path),
            method=request.method,
            status="success",
            details={
                "count": len(batch_response["results"]),
                "fraud_count": fraud_count,
                "approved_count": approved_count,
                "mongo_id": mongo_id,
            },
        )

        return batch_response

    except Exception as exc:
        await log_request_event(
            client_name=client["client_name"],
            endpoint=str(request.url.path),
            method=request.method,
            status="error",
            details={"error": str(exc)},
        )
        logger.error(f"Batch Prediction Error: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(exc)}")


@router.post("/report")
async def report(
    req: ReportRequest,
    request: Request,
    client=Depends(verify_api_key),
):
    # Build a more detailed report card for one transaction.
    _ensure_models_ready()

    try:
        card = build_report_response(req, client_name=client["client_name"])

        mongo_id = await save_report_to_mongo(req, card, client_name=client["client_name"])
        card["mongo_id"] = mongo_id

        await log_activity(
            action="report_generated",
            actor=client["client_name"],
            details={
                "transaction_id": req.transaction_id,
                "mongo_id": mongo_id,
                "decision": card["decision"],
            },
        )

        await log_request_event(
            client_name=client["client_name"],
            endpoint=str(request.url.path),
            method=request.method,
            status="success",
            details={
                "transaction_id": req.transaction_id,
                "mongo_id": mongo_id,
                "decision": card["decision"],
            },
        )

        return card

    except Exception as exc:
        await log_request_event(
            client_name=client["client_name"],
            endpoint=str(request.url.path),
            method=request.method,
            status="error",
            details={"error": str(exc), "transaction_id": req.transaction_id},
        )
        logger.error(f"Report Error: {str(exc)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(exc)}")
