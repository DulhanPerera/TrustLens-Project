from datetime import timedelta

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException

try:
    from core.settings import API_KEY_DEFAULT_EXPIRY_DAYS
    from core.state import state
    from schemas.request_models import ApiKeyCreateRequest, ApiKeyStatusUpdateRequest
    from services.auth_service import (
        build_token_preview,
        generate_api_token,
        hash_token,
        utcnow,
        verify_admin_setup_key,
    )
    from services.mongo_service import log_activity, serialize_doc
except ImportError:
    from backend.core.settings import API_KEY_DEFAULT_EXPIRY_DAYS
    from backend.core.state import state
    from backend.schemas.request_models import ApiKeyCreateRequest, ApiKeyStatusUpdateRequest
    from backend.services.auth_service import (
        build_token_preview,
        generate_api_token,
        hash_token,
        utcnow,
        verify_admin_setup_key,
    )
    from backend.services.mongo_service import log_activity, serialize_doc


router = APIRouter()


@router.post("/api-keys")
async def create_api_key(
    payload: ApiKeyCreateRequest,
    _admin_ok: bool = Depends(verify_admin_setup_key),
):
    if state.mongo_api_keys_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB API key collection is not available")

    raw_token = generate_api_token()
    token_hash = hash_token(raw_token)
    token_preview = build_token_preview(raw_token)
    now = utcnow()
    expires_at = now + timedelta(days=API_KEY_DEFAULT_EXPIRY_DAYS)

    doc = {
        "client_name": payload.client_name.strip(),
        "key_hash": token_hash,
        "token_preview": token_preview,
        "status": "active",
        "created_at": now,
        "expires_at": expires_at,
        "last_used_at": None,
        "shown_once": True,
    }

    result = await state.mongo_api_keys_collection.insert_one(doc)

    await log_activity(
        action="api_key_created",
        actor="admin",
        details={
            "client_name": payload.client_name,
            "api_key_id": str(result.inserted_id),
            "expires_at": expires_at.isoformat(),
            "token_preview": token_preview,
        },
    )

    return {
        "message": "API key created successfully. Copy and store this token now; it will not be shown again.",
        "api_key_id": str(result.inserted_id),
        "client_name": payload.client_name,
        "status": "active",
        "expires_in_days": API_KEY_DEFAULT_EXPIRY_DAYS,
        "expires_at": expires_at.isoformat(),
        "token_preview": token_preview,
        "token": raw_token,
    }


@router.get("/api-keys")
async def list_api_keys(
    limit: int = 50,
    _admin_ok: bool = Depends(verify_admin_setup_key),
):
    if state.mongo_api_keys_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB API key collection is not available")

    limit = max(1, min(limit, 200))
    cursor = state.mongo_api_keys_collection.find({}, {"key_hash": 0}).sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)

    return {"count": len(docs), "items": [serialize_doc(doc) for doc in docs]}


@router.get("/api-keys/{api_key_id}")
async def get_api_key_by_id(
    api_key_id: str,
    _admin_ok: bool = Depends(verify_admin_setup_key),
):
    if state.mongo_api_keys_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB API key collection is not available")

    try:
        doc = await state.mongo_api_keys_collection.find_one({"_id": ObjectId(api_key_id)}, {"key_hash": 0})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid API key id")

    if not doc:
        raise HTTPException(status_code=404, detail="API key not found")

    return serialize_doc(doc)


@router.patch("/api-keys/{api_key_id}/status")
async def update_api_key_status(
    api_key_id: str,
    payload: ApiKeyStatusUpdateRequest,
    _admin_ok: bool = Depends(verify_admin_setup_key),
):
    if state.mongo_api_keys_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB API key collection is not available")

    try:
        result = await state.mongo_api_keys_collection.update_one(
            {"_id": ObjectId(api_key_id)},
            {"$set": {"status": payload.status, "updated_at": utcnow()}},
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid API key id")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="API key not found")

    updated_doc = await state.mongo_api_keys_collection.find_one({"_id": ObjectId(api_key_id)}, {"key_hash": 0})

    await log_activity(
        action="api_key_status_updated",
        actor="admin",
        details={"api_key_id": api_key_id, "status": payload.status},
    )

    return {
        "message": "API key status updated successfully",
        "item": serialize_doc(updated_doc),
    }


@router.get("/request-logs")
async def get_request_logs(
    limit: int = 50,
    _admin_ok: bool = Depends(verify_admin_setup_key),
):
    if state.mongo_request_log_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB request log collection is not available")

    limit = max(1, min(limit, 200))
    cursor = state.mongo_request_log_collection.find().sort("created_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)

    return {"count": len(docs), "items": [serialize_doc(doc) for doc in docs]}
