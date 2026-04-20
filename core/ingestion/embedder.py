# core/ingestion/embedder.py
# ============================================================
# Embedding model — uses langchain-huggingface (new package).
#
# Install: pip install -U langchain-huggingface
#
# BGE query prefix is handled via a thin wrapper class below
# because HuggingFaceEmbeddings no longer accepts query_instruction.
# ============================================================

from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings
from config.settings import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# BGE models need this prefix at query time (not at indexing time)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BGEEmbeddings(Embeddings):
    """
    Thin wrapper around HuggingFaceEmbeddings that applies the BGE
    query prefix only during embed_query() — not during embed_documents().
    This is the correct usage per the BGE paper and gives 5-10% better
    retrieval quality vs using the same text for both.
    """

    def __init__(self, model: "HuggingFaceEmbeddings", is_bge: bool):
        self._model = model
        self._is_bge = is_bge

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Index-time: plain text, no prefix."""
        return self._model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Query-time: prepend BGE instruction prefix."""
        if self._is_bge:
            text = _BGE_QUERY_PREFIX + text
        return self._model.embed_query(text)


@lru_cache(maxsize=1)
def get_embedding_model() -> Embeddings:
    """
    Returns a cached BGEEmbeddings instance.
    First call downloads the model (~100-340MB) from HuggingFace Hub.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = settings.embedding_model
    device = settings.embedding_device
    is_bge = "bge" in model_name.lower()

    logger.info(
        f"🔌 Embedding model: [bold]{model_name}[/bold] "
        f"| device={device} "
        f"| bge_prefix={is_bge}"
    )

    base = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={
            "device": device,
            "trust_remote_code": False,
        },
        encode_kwargs={
            "normalize_embeddings": True,  # unit vectors → cosine sim = dot product
            "batch_size": 32,
        },
    )

    logger.info("  ✅ Embedding model loaded.")
    return BGEEmbeddings(model=base, is_bge=is_bge)