"""
This file keeps shared runtime values for the backend.
The app stores loaded models and Mongo collections here.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RuntimeState:
    # These values get filled in when the app starts.
    mlp_model: Optional[Any] = None
    ae_model: Optional[Any] = None
    scaler: Optional[Any] = None
    calibrator: Optional[Any] = None
    thresholds: Dict[str, Any] = field(default_factory=dict)
    shap_explainer: Optional[Any] = None
    feature_names: List[str] = field(
        default_factory=lambda: ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    )
    last_init_error: Optional[str] = None
    mongo_client: Optional[Any] = None
    mongo_db: Optional[Any] = None
    mongo_tx_collection: Optional[Any] = None
    mongo_report_collection: Optional[Any] = None
    mongo_user_collection: Optional[Any] = None
    mongo_settings_collection: Optional[Any] = None
    mongo_activity_collection: Optional[Any] = None
    mongo_api_keys_collection: Optional[Any] = None
    mongo_request_log_collection: Optional[Any] = None

# One shared state object is enough for the whole backend process.
state = RuntimeState()
