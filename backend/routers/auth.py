"""
Auth routes live here.
They handle Google sign-in and simple user role updates.
"""

from bson import ObjectId
from fastapi import APIRouter, HTTPException

try:
    from core.state import state
    from schemas.request_models import GoogleAuthRequest, UpdateUserRoleRequest
    from services.auth_service import authenticate_google_user, normalize_role
    from services.mongo_service import log_activity, serialize_doc
    from utils.logger import get_logger
except ImportError:
    from backend.core.state import state
    from backend.schemas.request_models import GoogleAuthRequest, UpdateUserRoleRequest
    from backend.services.auth_service import authenticate_google_user, normalize_role
    from backend.services.mongo_service import log_activity, serialize_doc
    from backend.utils.logger import get_logger


router = APIRouter()
logger = get_logger(__name__)


@router.post("/auth/google")
async def auth_google(payload: GoogleAuthRequest):
    # Pass the Google token to the auth service and return the result.
    return await authenticate_google_user(payload)


@router.get("/users")
async def get_users(limit: int = 20):
    if state.mongo_user_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB user collection is not available")

    limit = max(1, min(limit, 200))
    cursor = state.mongo_user_collection.find().sort("last_login_at", -1).limit(limit)
    docs = await cursor.to_list(length=limit)

    return {"count": len(docs), "items": [serialize_doc(doc) for doc in docs]}


@router.patch("/users/{user_id}/role")
async def update_user_role(user_id: str, payload: UpdateUserRoleRequest):
    # Keep roles limited to the allowed values from settings.
    if state.mongo_user_collection is None:
        raise HTTPException(status_code=503, detail="MongoDB user collection is not available")

    new_role = normalize_role(payload.role)

    try:
        result = await state.mongo_user_collection.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"role": new_role}},
        )
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user id")

    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="User not found")

    updated_user = await state.mongo_user_collection.find_one({"_id": ObjectId(user_id)})

    await log_activity(
        action="user_role_updated",
        actor="admin",
        details={"user_id": user_id, "new_role": new_role},
    )

    logger.info(f"user_role_updated: user_id={user_id} new_role={new_role}")

    return {
        "message": "User role updated successfully",
        "user": serialize_doc(updated_user),
    }
