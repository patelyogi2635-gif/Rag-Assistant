# utils/logger.py
# ============================================================
# Structured, colored logging for the whole application.
# Usage: from utils.logger import get_logger
#        logger = get_logger(__name__)
# ============================================================

import logging
import sys
from rich.logging import RichHandler
from config.settings import get_settings


def get_logger(name: str) -> logging.Logger:
    settings = get_settings()

    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                rich_tracebacks=True,
                show_path=False,
                markup=True,
            )
        ],
        force=True,
    )
    return logging.getLogger(name)