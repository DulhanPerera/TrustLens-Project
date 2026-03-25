from typing import Any, List, Optional

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    transaction_id: Optional[str] = None
    Time: float
    V_features: List[float] = Field(..., min_items=28, max_items=28)
    Amount: float
    include_xai: bool = True
    top_k: int = Field(4, ge=1, le=15)


class BatchRequest(BaseModel):
    transactions: List[Transaction] = Field(..., min_items=50, max_items=500)
    include_xai: bool = False
    top_k: int = Field(4, ge=1, le=15)


class ReportRequest(BaseModel):
    transaction_id: Optional[str] = None
    Time: float
    V_features: List[float] = Field(..., min_items=28, max_items=28)
    Amount: float
    top_k: int = Field(6, ge=1, le=15)
    include_shap: bool = True
    include_recon: bool = True
    include_raw_features: bool = False


class GoogleAuthRequest(BaseModel):
    credential: str
    login_type: str = "dashboard"


class UpdateUserRoleRequest(BaseModel):
    role: str


class AnalystNoteRequest(BaseModel):
    note: str


class SettingUpdate(BaseModel):
    key: str
    value: Any


class ApiKeyCreateRequest(BaseModel):
    client_name: str = Field(..., min_length=2, max_length=120)


class ApiKeyStatusUpdateRequest(BaseModel):
    status: str = Field(..., pattern="^(active|inactive)$")
