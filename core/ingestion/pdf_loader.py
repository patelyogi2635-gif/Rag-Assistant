# core/ingestion/pdf_loader.py
# ============================================================
# PDF → raw text extraction.
# Strategy: try pdfplumber first (handles tables/complex layouts),
#           fall back to pypdf (faster, simpler).
# ============================================================

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pdfplumber
from pypdf import PdfReader

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RawPage:
    """Text content of a single PDF page."""
    page_number: int        # 1-indexed
    text: str
    char_count: int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) == 0


@dataclass
class RawDocument:
    """All pages extracted from a single PDF file."""
    filename: str
    file_path: Path
    pages: List[RawPage]

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def non_empty_pages(self) -> List[RawPage]:
        return [p for p in self.pages if not p.is_empty]

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.non_empty_pages)


class PDFLoader:
    """
    Loads PDF files and extracts text page-by-page.

    Usage:
        loader = PDFLoader()
        doc = loader.load(Path("my_file.pdf"))
    """

    def load(self, path: Path) -> RawDocument:
        """
        Load a single PDF. Tries pdfplumber first, falls back to pypdf.
        """
        logger.info(f"Loading PDF: [bold]{path.name}[/bold]")

        pages = self._extract_with_pdfplumber(path)

        # If pdfplumber got nothing useful, try pypdf
        total_text = sum(len(p.text.strip()) for p in pages)
        if total_text < 100:
            logger.warning(
                f"pdfplumber extracted little text from '{path.name}'. "
                "Falling back to pypdf."
            )
            pages = self._extract_with_pypdf(path)

        doc = RawDocument(
            filename=path.name,
            file_path=path,
            pages=pages,
        )

        logger.info(
            f"  → {doc.total_pages} pages | "
            f"{len(doc.non_empty_pages)} non-empty | "
            f"{len(doc.full_text):,} chars"
        )
        return doc

    def load_many(self, paths: List[Path]) -> List[RawDocument]:
        """Load multiple PDFs, collecting results even if one fails."""
        results = []
        for path in paths:
            try:
                results.append(self.load(path))
            except Exception as e:
                logger.error(f"Failed to load '{path.name}': {e}")
        return results

    # ── Private Extraction Methods ───────────────────────────

    def _extract_with_pdfplumber(self, path: Path) -> List[RawPage]:
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                text = self._clean_text(text)
                pages.append(RawPage(page_number=i, text=text))
        return pages

    def _extract_with_pypdf(self, path: Path) -> List[RawPage]:
        pages = []
        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = self._clean_text(text)
            pages.append(RawPage(page_number=i, text=text))
        return pages

    def _clean_text(self, text: str) -> str:
        """Basic cleanup — collapse excessive whitespace."""
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)   # max 2 consecutive newlines
        text = re.sub(r" {2,}", " ", text)         # collapse spaces
        return text.strip()