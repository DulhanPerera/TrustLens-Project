"""
TrustLens FastAPI API
(MLP + Autoencoder + optional SHAP + MongoDB + Google Auth + Role-Based Access + API Key Management + .env)

Run from `backend/`:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

try:
    from core.settings import CORS_ORIGINS
    from routers.admin import router as admin_router
    from routers.api_keys import router as api_keys_router
    from routers.auth import router as auth_router
    from routers.predictions import router as predictions_router
    from services.mongo_service import close_mongo, init_mongo
    from services.prediction_service import load_assets
    from utils.logger import setup_logging
except ImportError:
    from backend.core.settings import CORS_ORIGINS
    from backend.routers.admin import router as admin_router
    from backend.routers.api_keys import router as api_keys_router
    from backend.routers.auth import router as auth_router
    from backend.routers.predictions import router as predictions_router
    from backend.services.mongo_service import close_mongo, init_mongo
    from backend.services.prediction_service import load_assets
    from backend.utils.logger import setup_logging


setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_mongo()
    load_assets()
    yield
    await close_mongo()


app = FastAPI(
    title="TrustLens API",
    description="Fraud Detection with MLP + Autoencoder + XAI (Explainable AI) + MongoDB + Google Auth + API Key Management",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_keys_router)
app.include_router(auth_router)
app.include_router(predictions_router)
app.include_router(admin_router)
