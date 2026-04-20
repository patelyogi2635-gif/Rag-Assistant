# core/ingestion/pipeline.py
# ============================================================
# Ingestion Pipeline — the public entry point for adding PDFs.
#
# This is the ONLY module callers should interact with for ingestion.
# It composes: validate → load → chunk → embed → store
# ============================================================

import time
from pathlib import Path
from typing import List

from config.settings import get_settings
from core.ingestion.chunker import SemanticChunker
from core.ingestion.pdf_loader import PDFLoader
from core.vectorstore.chroma_store import ChromaVectorStore
from models.schemas import BatchIngestionResult, IngestionResult
from utils.file_utils import FileValidationError, compute_file_hash, validate_pdf_files
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class IngestionPipeline:
    """
    Orchestrates the full PDF ingestion flow.

    Usage:
        pipeline = IngestionPipeline()
        result = pipeline.ingest([Path("medical_report.pdf")])
        print(result.total_chunks_added)
    """

    def __init__(self):
        self._loader = PDFLoader()
        self._chunker = SemanticChunker()
        self._store = ChromaVectorStore()

    def ingest(self, file_paths: List[Path]) -> BatchIngestionResult:
        """
        Ingest a batch of PDFs into the vector store.

        Steps:
          1. Validate all files upfront (fail fast).
          2. For each file, check for duplicates via hash.
          3. Load → chunk → embed → store.
          4. Return a detailed result report.
        """
        start = time.perf_counter()

        # Step 1: Validate (raises FileValidationError on failure)
        try:
            validate_pdf_files(file_paths)
        except FileValidationError as e:
            logger.error(f"Validation failed: {e}")
            raise

        results: List[IngestionResult] = []
        total_chunks_added = 0

        # Step 2-4: Process each file
        for path in file_paths:
            result = self._ingest_single(path)
            results.append(result)
            if not result.was_duplicate and result.error is None:
                total_chunks_added += result.total_chunks

        duration = round(time.perf_counter() - start, 3)

        batch_result = BatchIngestionResult(
            processed=results,
            total_chunks_added=total_chunks_added,
            duration_seconds=duration,
        )

        self._log_summary(batch_result)
        return batch_result

    # ── Private ──────────────────────────────────────────────

    def _ingest_single(self, path: Path) -> IngestionResult:
        """Process one PDF file. Returns an IngestionResult."""
        logger.info(f"\n📄 Processing: [bold]{path.name}[/bold]")

        file_hash = compute_file_hash(path)

        # Duplicate check
        if self._store.is_file_indexed(file_hash):
            logger.warning(f"  ⚠️  '{path.name}' already indexed. Skipping.")
            return IngestionResult(
                filename=path.name,
                file_hash=file_hash,
                total_pages=0,
                total_chunks=0,
                was_duplicate=True,
            )

        try:
            # Load
            raw_doc = self._loader.load(path)

            # Chunk
            chunks = self._chunker.chunk_document(raw_doc)

            # Embed + Store
            self._store.add_chunks(chunks)

            return IngestionResult(
                filename=path.name,
                file_hash=file_hash,
                total_pages=raw_doc.total_pages,
                total_chunks=len(chunks),
            )

        except Exception as e:
            logger.error(f"  ❌ Failed to process '{path.name}': {e}")
            return IngestionResult(
                filename=path.name,
                file_hash=file_hash,
                total_pages=0,
                total_chunks=0,
                error=str(e),
            )

    def _log_summary(self, result: BatchIngestionResult) -> None:
        logger.info("\n" + "=" * 50)
        logger.info("📊 Ingestion Summary")
        logger.info("=" * 50)
        for r in result.processed:
            status = (
                "⚠️  DUPLICATE" if r.was_duplicate
                else f"❌ ERROR: {r.error}" if r.error
                else f"✅ {r.total_chunks} chunks / {r.total_pages} pages"
            )
            logger.info(f"  {r.filename}: {status}")
        logger.info(
            f"\nTotal chunks added: {result.total_chunks_added} "
            f"in {result.duration_seconds}s"
        )