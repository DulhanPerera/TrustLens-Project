"""
This file sets up app logging.
It makes sure logs go to the console and the audit file.
"""

import logging
import logging.config
from pathlib import Path
from threading import Lock

try:
    from utils.core_config import get_core_config
except ImportError:
    from backend.utils.core_config import get_core_config


_CONFIG_LOCK = Lock()
_IS_CONFIGURED = False


def _logger_config_path() -> Path:
    return get_core_config().base_dir / "config" / "logger.ini"


def _log_file_path() -> Path:
    return get_core_config().trustlens_audit_log_path


def setup_logging() -> logging.Logger:
    global _IS_CONFIGURED

    with _CONFIG_LOCK:
        # Only configure logging once for the whole process.
        if _IS_CONFIGURED:
            return logging.getLogger("trustlens")

        log_file_path = _log_file_path()
        log_file_path.parent.mkdir(parents=True, exist_ok=True)

        logging.config.fileConfig(
            _logger_config_path(),
            defaults={"log_file": str(log_file_path)},
            disable_existing_loggers=False,
        )

        _IS_CONFIGURED = True
        return logging.getLogger("trustlens")


def get_logger(name: str | None = None) -> logging.Logger:
    # Keep logger names under the trustlens.* namespace.
    setup_logging()

    if not name:
        return logging.getLogger("trustlens")

    normalized_name = name.replace("backend.", "").strip(".")
    if normalized_name.startswith("trustlens"):
        return logging.getLogger(normalized_name)

    return logging.getLogger(f"trustlens.{normalized_name}")
