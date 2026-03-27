"""
This file reads the backend config file.
It gives the rest of the app one simple place to get paths from.
"""

import configparser
import os
from pathlib import Path
from typing import List, Optional


class CoreConfig:
    # Keep one shared config object so we do not reread the file everywhere.
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.base_dir = Path(__file__).resolve().parent.parent
        self.config_path = self.base_dir / "config" / "core_config.ini"
        self.config = configparser.ConfigParser()
        self.config.read(self.config_path)
        self._initialized = True

    def _config_path(
        self,
        section: str,
        option: str,
        default: Path,
        env_name: Optional[str] = None,
    ) -> Path:
        # Let env vars override the config file when needed.
        env_name = env_name or option.upper()
        raw_value = os.getenv(env_name) or self.config.get(section, option, fallback=str(default))
        candidate = Path(raw_value).expanduser()
        if candidate.is_absolute():
            return candidate
        return (self.base_dir / candidate).resolve()

    def _config_path_list(self, section: str, option: str, defaults: List[Path]) -> List[Path]:
        raw_value = self.config.get(section, option, fallback="")
        if not raw_value.strip():
            return defaults

        parsed_paths = []
        for item in raw_value.split(","):
            value = item.strip()
            if not value:
                continue
            candidate = Path(value).expanduser()
            parsed_paths.append(
                candidate if candidate.is_absolute() else (self.base_dir / candidate).resolve()
            )
        return parsed_paths or defaults

    @property
    def mlp_model_path(self) -> Path:
        return self._config_path("MODEL_PATHS", "mlp_model_path", self.base_dir / "mlp_model.keras")

    @property
    def ae_model_path(self) -> Path:
        return self._config_path("MODEL_PATHS", "ae_model_path", self.base_dir / "autoencoder.keras")

    @property
    def scaler_path(self) -> Path:
        return self._config_path("MODEL_PATHS", "scaler_path", self.base_dir / "scaler.joblib")

    @property
    def calibrator_path(self) -> Path:
        return self._config_path(
            "MODEL_PATHS",
            "calibrator_path",
            self.base_dir / "isotonic_calibrator.joblib",
        )

    @property
    def thresholds_path(self) -> Path:
        return self._config_path("MODEL_PATHS", "thresholds_path", self.base_dir / "thresholds.json")

    @property
    def alt_mlp_paths(self) -> List[Path]:
        return self._config_path_list(
            "MODEL_PATHS",
            "alt_mlp_paths",
            [self.base_dir / "mlp_best.keras", self.base_dir / "trustlens_model.keras"],
        )

    @property
    def alt_ae_paths(self) -> List[Path]:
        return self._config_path_list(
            "MODEL_PATHS",
            "alt_ae_paths",
            [self.base_dir / "ae_best.keras"],
        )

    @property
    def alt_scaler_paths(self) -> List[Path]:
        return self._config_path_list(
            "MODEL_PATHS",
            "alt_scaler_paths",
            [self.base_dir / "preprocess_artifacts.pkl"],
        )

    @property
    def trustlens_audit_log_path(self) -> Path:
        return self._config_path(
            "LOGGER_PATHS",
            "trustlens_audit",
            self.base_dir / "logs" / "trustlens_audit.log",
            env_name="TRUSTLENS_AUDIT_LOG_PATH",
        )


def get_core_config() -> CoreConfig:
    return CoreConfig()
