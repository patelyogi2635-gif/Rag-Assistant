# api/routers/documents.py
# ============================================================
# Document management endpoints:
#
#   POST   /documents/upload    Upload 1–5 PDFs, ingest into vector DB
#   GET    /documents/          List all indexed files
#   DELETE /documents/{filename} Remove a file from the vector DB
# ============================================================

import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from config.settings import get_settings
from core.ingestion.pipeline import IngestionPipeline
from core.vectorstore.chroma_store import ChromaVectorStore
from models.schemas import (
    BatchIngestionResult,
    DeleteFileResponse,
    IndexedFile,
    IndexedFilesResponse,
)
from utils.file_utils import FileValidationError
from utils.logger import get_logger

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = get_logger(__name__)
settings = get_settings()


@router.post(
    "/upload",
    response_model=BatchIngestionResult,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest PDFs",
    description=(
        f"Upload between 1 and {settings.max_pdf_files} PDF files. "
        "Each file is validated, chunked, embedded, and stored in ChromaDB. "
        "Re-uploading the same file is safe — it is detected via SHA-256 hash and skipped."
    ),
)
async def upload_documents(
    files: List[UploadFile] = File(..., description="PDF files to ingest (max 5)"),
) -> BatchIngestionResult:
    # Validate count before touching disk
    if len(files) > settings.max_pdf_files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many files. Maximum allowed: {settings.max_pdf_files}.",
        )

    # Save uploaded files to disk temporarily
    saved_paths: List[Path] = []
    try:
        for upload in files:
            if not upload.filename:
                raise HTTPException(400, "File has no filename.")

            dest = settings.upload_path / upload.filename
            with open(dest, "wb") as f:
                shutil.copyfileobj(upload.file, f)
            saved_paths.append(dest)
            logger.info(f"📥 Saved upload: {upload.filename}")

        # Run ingestion pipeline
        pipeline = IngestionPipeline()
        result = pipeline.ingest(saved_paths)
        return result

    except FileValidationError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.get(
    "/",
    response_model=IndexedFilesResponse,
    summary="List indexed files",
)
async def list_documents() -> IndexedFilesResponse:
    store = ChromaVectorStore()
    filenames = store.list_indexed_files()

    # Count chunks per file
    files = []
    for name in filenames:
        results = store._store.get(where={"source": {"$eq": name}})
        files.append(IndexedFile(filename=name, chunk_count=len(results["ids"])))

    return IndexedFilesResponse(
        files=files,
        total_files=len(files),
        total_chunks=store.total_documents,
    )


@router.delete(
    "/{filename}",
    response_model=DeleteFileResponse,
    summary="Remove a file from the vector store",
)
async def delete_document(filename: str) -> DeleteFileResponse:
    store = ChromaVectorStore()
    deleted = store.delete_file(filename)

    if deleted == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{filename}' not found in the vector store.",
        )

    return DeleteFileResponse(
        filename=filename,
        chunks_deleted=deleted,
        success=True,
    )