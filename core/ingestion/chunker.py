# core/ingestion/chunker.py
# ============================================================
# Text splitting strategy for RAG.
#
# Design decisions:
#   - RecursiveCharacterTextSplitter: respects sentence/paragraph
#     boundaries — produces semantically coherent chunks.
#   - chunk_size in TOKENS (via tiktoken), not characters.
#     This is critical: LLMs have token limits, not char limits.
#   - Each chunk carries full metadata (source, page, index)
#     so citations are possible in the final answer.
# ============================================================

import uuid
from typing import List

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import get_settings
from core.ingestion.pdf_loader import RawDocument, RawPage
from models.schemas import DocumentChunk
from utils.file_utils import compute_file_hash
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class SemanticChunker:
    """
    Splits a RawDocument into DocumentChunks suitable for embedding.

    Chunking is per-page so page-number metadata stays accurate.
    If a page's text is shorter than chunk_size, it becomes one chunk.
    """

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
        model_name: str = "cl100k_base",   # tokenizer used by OpenAI models
    ):
        self._tokenizer = tiktoken.get_encoding(model_name)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_len,  # measure in tokens, not chars
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: RawDocument) -> List[DocumentChunk]:
        """
        Chunk all pages of a document.
        Returns a flat list of DocumentChunk objects.
        """
        file_hash = compute_file_hash(doc.file_path)
        chunks: List[DocumentChunk] = []
        global_index = 0

        for page in doc.non_empty_pages:
            page_chunks = self._chunk_page(
                page=page,
                filename=doc.filename,
                file_hash=file_hash,
                start_index=global_index,
            )
            chunks.extend(page_chunks)
            global_index += len(page_chunks)

        logger.info(
            f"  → '{doc.filename}': {len(doc.non_empty_pages)} pages "
            f"→ {len(chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return chunks

    def chunk_documents(self, docs: List[RawDocument]) -> List[DocumentChunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(f"Total chunks across all documents: {len(all_chunks)}")
        return all_chunks

    # ── Private ──────────────────────────────────────────────

    def _chunk_page(
        self,
        page: RawPage,
        filename: str,
        file_hash: str,
        start_index: int,
    ) -> List[DocumentChunk]:
        """Split a single page into chunks and wrap in DocumentChunk."""
        raw_splits = self._splitter.split_text(page.text)
        chunks = []

        for i, text in enumerate(raw_splits):
            chunk = DocumentChunk(
                chunk_id=str(uuid.uuid4()),
                source_file=filename,
                file_hash=file_hash,
                page_number=page.page_number,
                chunk_index=start_index + i,
                content=text,
                token_count=self._token_len(text),
                metadata={
                    "source": filename,
                    "page": page.page_number,
                    "chunk_index": start_index + i,
                    "file_hash": file_hash,
                },
            )
            chunks.append(chunk)

        return chunks

    def _token_len(self, text: str) -> int:
        return len(self._tokenizer.encode(text))