"""
MongoDB helper logic lives here.
This file saves records, cleans data, and opens the database connection.
"""

from datetime import datetime
from typing import Any, Dict, Optional

from bson import ObjectId
from pymongo import AsyncMongoClient
from pymongo.server_api import ServerApi

try:
    from core.settings import (
        MONGO_ACTIVITY_COLLECTION,
        MONGO_API_KEYS_COLLECTION,
        MONGO_DB_NAME,
        MONGO_REPORT_COLLECTION,
        MONGO_REQUEST_LOG_COLLECTION,
        MONGO_SETTINGS_COLLECTION,
        MONGO_TX_COLLECTION,
        MONGO_URI,
        MONGO_USER_COLLECTION,
    )
    from core.state import state
    from utils.logger import get_logger
except ImportError:
    from backend.core.settings import (
        MONGO_ACTIVITY_COLLECTION,
        MONGO_API_KEYS_COLLECTION,
        MONGO_DB_NAME,
        MONGO_REPORT_COLLECTION,
        MONGO_REQUEST_LOG_COLLECTION,
        MONGO_SETTINGS_COLLECTION,
        MONGO_TX_COLLECTION,
        MONGO_URI,
        MONGO_USER_COLLECTION,
    )
    from backend.core.state import state
    from backend.utils.logger import get_logger


logger = get_logger(__name__)


def sanitize_for_mongo(obj: Any) -> Any:
    # Convert numpy values into normal Python values before saving.
    if isinstance(obj, dict):
        return {k: sanitize_for_mongo(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_mongo(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_mongo(v) for v in obj]
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
    except Exception:
        pass
    return obj


def serialize_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
    # Turn ObjectId and datetime values into JSON-friendly values.
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


async def log_activity(action: str, actor: str = "system", details: Optional[dict] = None):
    if state.mongo_activity_collection is None:
        return

    doc = {
        "action": action,
        "actor": actor,
        "details": details or {},
        "created_at": datetime.utcnow(),
    }
    await state.mongo_activity_collection.insert_one(doc)


async def log_request_event(
    client_name: str,
    endpoint: str,
    method: str,
    status: str,
    details: Optional[dict] = None,
):
    if state.mongo_request_log_collection is None:
        return

    doc = {
        "client_name": client_name,
        "endpoint": endpoint,
        "method": method,
        "status": status,
        "details": sanitize_for_mongo(details or {}),
        "created_at": datetime.utcnow(),
    }
    await state.mongo_request_log_collection.insert_one(doc)


async def save_prediction_to_mongo(
    tx,
    response_payload: Dict[str, Any],
    client_name: str = "unknown",
) -> Optional[str]:
    # Keep the original input and the model output together in one record.
    if state.mongo_tx_collection is None:
        return None

    doc = {
        "type": "predict",
        "client_name": client_name,
        "transaction_id": tx.transaction_id,
        "input": {
            "Time": tx.Time,
            "V_features": tx.V_features,
            "Amount": tx.Amount,
        },
        "output": sanitize_for_mongo(response_payload),
        "is_fraud": response_payload.get("is_fraud"),
        "status": response_payload.get("status"),
        "risk_score": response_payload.get("risk_score"),
        "created_at": datetime.utcnow(),
    }

    result = await state.mongo_tx_collection.insert_one(doc)
    return str(result.inserted_id)


async def save_report_to_mongo(
    req,
    report_payload: Dict[str, Any],
    client_name: str = "unknown",
) -> Optional[str]:
    if state.mongo_report_collection is None:
        return None

    doc = {
        "type": "report",
        "client_name": client_name,
        "transaction_id": req.transaction_id,
        "input": {
            "Time": req.Time,
            "V_features": req.V_features,
            "Amount": req.Amount,
        },
        "output": sanitize_for_mongo(report_payload),
        "decision": report_payload.get("decision"),
        "risk_score": report_payload.get("risk_score"),
        "created_at": datetime.utcnow(),
    }

    result = await state.mongo_report_collection.insert_one(doc)
    return str(result.inserted_id)


async def save_batch_to_mongo(
    req,
    batch_payload: Dict[str, Any],
    client_name: str = "unknown",
) -> Optional[str]:
    if state.mongo_tx_collection is None:
        return None

    created_at = datetime.utcnow()
    results = sanitize_for_mongo(batch_payload.get("results", []))
    thresholds = sanitize_for_mongo(batch_payload.get("thresholds", {}))
    fraud_count = sum(1 for item in results if item.get("is_fraud") is True)
    approved_count = sum(1 for item in results if item.get("is_fraud") is False)

    doc = {
        "type": "predict_batch",
        "client_name": client_name,
        "count": int(batch_payload.get("count", 0)),
        "fraud_count": fraud_count,
        "approved_count": approved_count,
        "sorted_by": batch_payload.get("sorted_by"),
        "thresholds": thresholds,
        "results": results,
        "created_at": created_at,
    }

    result = await state.mongo_tx_collection.insert_one(doc)
    batch_mongo_id = str(result.inserted_id)

    item_docs = []
    txs = list(getattr(req, "transactions", []) or [])

    for item in results:
        tx_index = item.get("index")
        tx = txs[tx_index] if isinstance(tx_index, int) and 0 <= tx_index < len(txs) else None
        input_payload = {
            "Time": getattr(tx, "Time", None),
            "V_features": list(getattr(tx, "V_features", []) or []),
            "Amount": getattr(tx, "Amount", None),
        }

        item_output = dict(item)
        item_output["thresholds"] = thresholds
        item_output["batch_mongo_id"] = batch_mongo_id

        item_docs.append(
            {
                "type": "predict_batch_item",
                "client_name": client_name,
                "batch_mongo_id": batch_mongo_id,
                "batch_index": tx_index if isinstance(tx_index, int) else None,
                "transaction_id": getattr(tx, "transaction_id", None) or item.get("transaction_id"),
                "input": sanitize_for_mongo(input_payload),
                "output": sanitize_for_mongo(item_output),
                "is_fraud": item.get("is_fraud"),
                "status": item.get("status"),
                "risk_score": item.get("risk_score"),
                "created_at": created_at,
            }
        )

    if item_docs:
        insert_many_result = await state.mongo_tx_collection.insert_many(item_docs)
        await state.mongo_tx_collection.update_one(
            {"_id": result.inserted_id},
            {
                "$set": {
                    "item_ids": [str(item_id) for item_id in insert_many_result.inserted_ids],
                }
            },
        )

    return batch_mongo_id


async def init_mongo():
    try:
        # Open the database and cache the collections the app uses often.
        state.mongo_client = AsyncMongoClient(
            MONGO_URI,
            server_api=ServerApi("1"),
        )
        state.mongo_db = state.mongo_client[MONGO_DB_NAME]
        state.mongo_tx_collection = state.mongo_db[MONGO_TX_COLLECTION]
        state.mongo_report_collection = state.mongo_db[MONGO_REPORT_COLLECTION]
        state.mongo_user_collection = state.mongo_db[MONGO_USER_COLLECTION]
        state.mongo_settings_collection = state.mongo_db[MONGO_SETTINGS_COLLECTION]
        state.mongo_activity_collection = state.mongo_db[MONGO_ACTIVITY_COLLECTION]
        state.mongo_api_keys_collection = state.mongo_db[MONGO_API_KEYS_COLLECTION]
        state.mongo_request_log_collection = state.mongo_db[MONGO_REQUEST_LOG_COLLECTION]

        await state.mongo_client.admin.command("ping")

        await state.mongo_tx_collection.create_index("created_at")
        await state.mongo_tx_collection.create_index("status")
        await state.mongo_tx_collection.create_index("is_fraud")
        await state.mongo_tx_collection.create_index("transaction_id")
        await state.mongo_tx_collection.create_index("type")
        await state.mongo_tx_collection.create_index("batch_mongo_id")

        await state.mongo_report_collection.create_index("created_at")
        await state.mongo_report_collection.create_index("decision")
        await state.mongo_report_collection.create_index("transaction_id")

        await state.mongo_user_collection.create_index("google_sub", unique=True)
        await state.mongo_user_collection.create_index("email")

        await state.mongo_activity_collection.create_index("created_at")
        await state.mongo_activity_collection.create_index("action")
        await state.mongo_settings_collection.create_index("key", unique=True)

        await state.mongo_api_keys_collection.create_index("client_name")
        await state.mongo_api_keys_collection.create_index("key_hash", unique=True)
        await state.mongo_api_keys_collection.create_index("status")
        await state.mongo_api_keys_collection.create_index("expires_at")

        await state.mongo_request_log_collection.create_index("created_at")
        await state.mongo_request_log_collection.create_index("client_name")
        await state.mongo_request_log_collection.create_index("endpoint")

        logger.info("MongoDB connected successfully.")

    except Exception as e:
        logger.exception(f"MongoDB initialization failed: {e}")
        state.mongo_client = None
        state.mongo_db = None
        state.mongo_tx_collection = None
        state.mongo_report_collection = None
        state.mongo_user_collection = None
        state.mongo_settings_collection = None
        state.mongo_activity_collection = None
        state.mongo_api_keys_collection = None
        state.mongo_request_log_collection = None


async def close_mongo():
    if state.mongo_client is not None:
        await state.mongo_client.close()
        logger.info("MongoDB connection closed.")
