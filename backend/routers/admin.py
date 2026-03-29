"""
Admin routes live here.
They handle health info, saved data, notes, and small admin actions.
"""

from datetime import datetime
from typing import Literal

from bson import ObjectId
from fastapi import APIRouter, HTTPException, Query
from pymongo.collation import Collation

try:
    from core.settings import (
        ADMIN_SETUP_KEY,
        API_KEY_DEFAULT_EXPIRY_DAYS,
        GOOGLE_CLIENT_ID,
        MONGO_ACTIVITY_COLLECTION,
        MONGO_API_KEYS_COLLECTION,
        MONGO_DB_NAME,
        MONGO_REPORT_COLLECTION,
        MONGO_REQUEST_LOG_COLLECTION,
        MONGO_SETTINGS_COLLECTION,
        MONGO_TX_COLLECTION,
        MONGO_USER_COLLECTION,
    )
    from core.state import state
    from schemas.request_models import AnalystNoteRequest, SettingUpdate
    from services.mongo_service import log_activity, serialize_doc
    from services.prediction_service import build_health_payload
except ImportError:
    from backend.core.settings import (
        ADMIN_SETUP_KEY,
        API_KEY_DEFAULT_EXPIRY_DAYS,
        GOOGLE_CLIENT_ID,
        MONGO_ACTIVITY_COLLECTION,
        MONGO_API_KEYS_COLLECTION,
        MONGO_DB_NAME,
        MONGO_REPORT_COLLECTION,
        MONGO_REQUEST_LOG_COLLECTION,
        MONGO_SETTINGS_COLLECTION,
        MONGO_TX_COLLECTION,
        MONGO_USER_COLLECTION,
    )
    from backend.core.state import state
    from backend.schemas.request_models import AnalystNoteRequest, SettingUpdate
    from backend.services.mongo_service import log_activity, serialize_doc
    from backend.services.prediction_service import build_health_payload


router = APIRouter()
TX_ID_COLLATION = Collation(locale="en", numericOrdering=True)


@router.patch("/transactions/{tx_id}/mark-legitimate")
async def mark_legitimate(tx_id: str):
    # Let an analyst override a flagged transaction.
    if state.mongo_tx_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    try:
        result = await state.mongo_tx_collection.update_one(
            {"_id": ObjectId(tx_id)},
            {
                "$set": {
                    "status": "APPROVED",
                    "is_fraud": False,
                    "analyst_override": True,
                }
            },
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid transaction id")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Transaction not found")

    updated_doc = await state.mongo_tx_collection.find_one({"_id": ObjectId(tx_id)})

    await log_activity("mark_legitimate", "analyst", {"transaction_id": tx_id})

    return {
        "message": "Transaction marked as legitimate",
        "item": serialize_doc(updated_doc),
    }


@router.post("/transactions/{tx_id}/note")
async def add_analyst_note(tx_id: str, payload: AnalystNoteRequest):
    # Save a quick note directly on the transaction record.
    if state.mongo_tx_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    note_doc = {
        "note": payload.note,
        "created_at": datetime.utcnow(),
    }

    try:
        result = await state.mongo_tx_collection.update_one(
            {"_id": ObjectId(tx_id)},
            {"$push": {"analyst_notes": note_doc}},
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid transaction id")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Transaction not found")

    updated_doc = await state.mongo_tx_collection.find_one({"_id": ObjectId(tx_id)})

    await log_activity(
        "analyst_note_added",
        "analyst",
        {"transaction_id": tx_id, "note": payload.note},
    )

    return {
        "message": "Note added",
        "item": serialize_doc(updated_doc),
    }


@router.get("/settings")
async def get_settings():
    if state.mongo_settings_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    cursor = state.mongo_settings_collection.find()
    docs = await cursor.to_list(length=100)

    return {"items": [serialize_doc(doc) for doc in docs]}


@router.post("/settings")
async def update_setting(payload: SettingUpdate):
    if state.mongo_settings_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    await state.mongo_settings_collection.update_one(
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


@router.get("/activity-logs")
async def get_activity_logs(limit: int = Query(default=200, ge=1, le=1000)):
    if state.mongo_activity_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    cursor = state.mongo_activity_collection.find().sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)

    return {"count": len(docs), "items": [serialize_doc(doc) for doc in docs]}


@router.get("/health")
def health():
    # Return a simple snapshot of what parts of the system are ready.
    return build_health_payload(
        mongo_config={
            "db_name": MONGO_DB_NAME,
            "tx_collection": MONGO_TX_COLLECTION,
            "report_collection": MONGO_REPORT_COLLECTION,
            "user_collection": MONGO_USER_COLLECTION,
            "settings_collection": MONGO_SETTINGS_COLLECTION,
            "activity_collection": MONGO_ACTIVITY_COLLECTION,
            "api_keys_collection": MONGO_API_KEYS_COLLECTION,
            "request_log_collection": MONGO_REQUEST_LOG_COLLECTION,
        },
        api_key_default_expiry_days=API_KEY_DEFAULT_EXPIRY_DAYS,
        google_client_id=GOOGLE_CLIENT_ID,
        admin_setup_key=ADMIN_SETUP_KEY,
    )


@router.get("/transactions")
async def get_transactions(
    limit: int = Query(default=200, ge=1, le=1000),
    sort_by: Literal["created_at", "transaction_id"] = "created_at",
    sort_dir: Literal["asc", "desc"] = "desc",
):
    if state.mongo_tx_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    direction = 1 if sort_dir == "asc" else -1
    query = {"type": {"$ne": "predict_batch"}}

    if sort_by == "transaction_id":
        cursor = (
            state.mongo_tx_collection.find(query, collation=TX_ID_COLLATION)
            .sort([("transaction_id", direction), ("created_at", -1)])
            .limit(limit)
        )
    else:
        cursor = state.mongo_tx_collection.find(query).sort("created_at", direction).limit(limit)

    docs = await cursor.to_list(length=limit)

    return {"count": len(docs), "items": [serialize_doc(doc) for doc in docs]}


@router.get("/reports")
async def get_reports(limit: int = 20):
    if state.mongo_report_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    limit = max(1, min(limit, 200))
    cursor = state.mongo_report_collection.find().sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)

    return {"count": len(docs), "items": [serialize_doc(doc) for doc in docs]}


@router.get("/transactions/{mongo_id}")
async def get_transaction_by_id(mongo_id: str):
    if state.mongo_tx_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    try:
        doc = await state.mongo_tx_collection.find_one({"_id": ObjectId(mongo_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid MongoDB ObjectId")

    if not doc:
        raise HTTPException(status_code=404, detail="Transaction not found")

    return serialize_doc(doc)


@router.get("/reports/{mongo_id}")
async def get_report_by_id(mongo_id: str):
    if state.mongo_report_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB not connected")

    try:
        doc = await state.mongo_report_collection.find_one({"_id": ObjectId(mongo_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid MongoDB ObjectId")

    if not doc:
        raise HTTPException(status_code=404, detail="Report not found")

    return serialize_doc(doc)
