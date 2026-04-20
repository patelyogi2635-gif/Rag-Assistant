# utils/file_utils.py
# ============================================================
# PDF file validation — called before any processing begins.
# Keeps ingestion layer clean of validation concerns.
# ============================================================

import hashlib
from pathlib import Path
from typing import List

from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class FileValidationError(Exception):
    """Raised when uploaded files fail validation."""
    pass


def validate_pdf_files(file_paths: List[Path]) -> None:
    """
    Validate a batch of PDFs before processing.
    Raises FileValidationError with a clear message on failure.
    """
    if not file_paths:
        raise FileValidationError("No files provided.")

    if len(file_paths) > settings.max_pdf_files:
        raise FileValidationError(
            f"Too many files. Maximum allowed: {settings.max_pdf_files}, "
            f"received: {len(file_paths)}."
        )

    for path in file_paths:
        _validate_single_file(path)

    logger.info(f"✅ Validated {len(file_paths)} PDF(s) successfully.")


def _validate_single_file(path: Path) -> None:
    """Validate extension, existence, size, and PDF magic bytes."""
    if not path.exists():
        raise FileValidationError(f"File not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise FileValidationError(
            f"Invalid file type '{path.suffix}' for '{path.name}'. Only PDFs allowed."
        )

    size = path.stat().st_size
    if size > settings.max_pdf_size_bytes:
        mb = size / (1024 * 1024)
        raise FileValidationError(
            f"'{path.name}' is {mb:.1f}MB, exceeds {settings.max_pdf_size_mb}MB limit."
        )

    if size == 0:
        raise FileValidationError(f"'{path.name}' is empty.")

    # Check PDF magic bytes (%PDF-)
    with open(path, "rb") as f:
        header = f.read(5)
    if header != b"%PDF-":
        raise FileValidationError(
            f"'{path.name}' does not appear to be a valid PDF file."
        )


def compute_file_hash(path: Path) -> str:
    """
    SHA-256 hash of file content.
    Used to detect duplicate uploads — same hash = already in vector DB.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()