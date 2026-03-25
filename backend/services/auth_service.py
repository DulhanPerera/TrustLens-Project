import hashlib
import secrets
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, Header, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token

try:
    from core.settings import (
        ADMIN_SETUP_KEY,
        ALLOWED_ADMIN_ROLES,
        ALLOWED_USER_ROLES,
        GOOGLE_CLIENT_ID,
    )
    from core.state import state
    from services.mongo_service import log_activity
    from utils.logger import get_logger
except ImportError:
    from backend.core.settings import (
        ADMIN_SETUP_KEY,
        ALLOWED_ADMIN_ROLES,
        ALLOWED_USER_ROLES,
        GOOGLE_CLIENT_ID,
    )
    from backend.core.state import state
    from backend.services.mongo_service import log_activity
    from backend.utils.logger import get_logger


bearer_scheme = HTTPBearer(auto_error=False)
logger = get_logger(__name__)


def generate_api_token() -> str:
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def build_token_preview(token: str) -> str:
    if len(token) <= 12:
        return token[:4] + "..."
    return f"{token[:6]}...{token[-4:]}"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_role(role: Optional[str]) -> str:
    role = (role or "user").strip().lower()
    if role not in ALLOWED_USER_ROLES:
        return "user"
    return role


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    if state.mongo_api_keys_collection is None:
        raise HTTPException(status_code=503, detail="API key store is not available")

    if credentials is None:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    if credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization scheme")

    raw_token = credentials.credentials
    key_hash = hash_token(raw_token)

    key_doc = await state.mongo_api_keys_collection.find_one({"key_hash": key_hash})

    if not key_doc:
        raise HTTPException(status_code=403, detail="Invalid API key")

    if key_doc.get("status") != "active":
        raise HTTPException(status_code=403, detail="API key is inactive")

    expires_at = key_doc.get("expires_at")
    if isinstance(expires_at, datetime):
        expires_at_cmp = expires_at
        if expires_at_cmp.tzinfo is None:
            expires_at_cmp = expires_at_cmp.replace(tzinfo=timezone.utc)
        if expires_at_cmp < utcnow():
            raise HTTPException(status_code=403, detail="API key has expired")

    await state.mongo_api_keys_collection.update_one(
        {"_id": key_doc["_id"]},
        {"$set": {"last_used_at": utcnow()}},
    )

    return {
        "key_id": str(key_doc["_id"]),
        "client_name": key_doc.get("client_name", "unknown"),
        "status": key_doc.get("status", "unknown"),
        "token_preview": key_doc.get("token_preview"),
    }


async def verify_admin_setup_key(x_admin_setup_key: Optional[str] = Header(default=None)):
    if not x_admin_setup_key:
        raise HTTPException(status_code=401, detail="Missing X-Admin-Setup-Key header")
    if x_admin_setup_key != ADMIN_SETUP_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin setup key")
    return True


async def authenticate_google_user(payload):
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=500, detail="GOOGLE_CLIENT_ID is not configured")

    if state.mongo_user_collection is None:
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

        existing_user = await state.mongo_user_collection.find_one({"google_sub": google_sub})
        role = normalize_role(existing_user.get("role")) if existing_user else "user"

        if login_type == "admin" and role not in ALLOWED_ADMIN_ROLES:
            raise HTTPException(
                status_code=403,
                detail="Access denied. Only admins and analysts can access the admin panel.",
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

        await state.mongo_user_collection.update_one(
            {"google_sub": google_sub},
            {"$set": user_doc},
            upsert=True,
        )

        await log_activity(
            action="google_login",
            actor=email,
            details={"google_sub": google_sub, "role": role, "login_type": login_type},
        )

        logger.info(
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
        logger.exception(f"Google auth failed: {e}")
        raise HTTPException(status_code=500, detail=f"Google auth failed: {str(e)}")
