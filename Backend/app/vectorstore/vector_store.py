import os
import logging
from math import ceil
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from app.vectorstore.embedding import get_embedding_model
from app.Config.config import settings

logger = logging.getLogger(__name__)

def _store_path_from_id(doc_id: str) -> str:
    return os.path.join(settings.VECTOR_DB_DIR, doc_id)

def create_vector_store(doc_id: str) -> Chroma:
    """
    Load a specific Chroma vector store by doc_id.
    """
    store_path = _store_path_from_id(doc_id)
    if not os.path.isdir(store_path):
        raise ValueError(f"Vector store not found at {store_path}")
    embedding = get_embedding_model()
    return Chroma(persist_directory=store_path, embedding_function=embedding)

def multi_similarity_search_with_score(
    doc_ids: List[str],
    query: str,
    max_chunks: int = 8,
) -> List[Tuple[Document, float]]:
    """
    Query multiple per document Chroma stores, gather candidates with scores,
    merge and return up to max_chunks best results.
    """
    if not doc_ids:
        return []

    per_store_k = max(2, ceil(max_chunks * 2 / max(len(doc_ids), 1)))
    candidates: List[Tuple[Document, float]] = []

    for d in doc_ids:
        try:
            vs = create_vector_store(d)
            # Chroma returns (Document, score) where lower distance is better
            hits = vs.similarity_search_with_score(query, k=per_store_k)
            candidates.extend(hits)
        except Exception as e:
            logger.warning(f"Search failed for {d}: {e}")

    # Sort by score ascending, keep top max_chunks
    candidates.sort(key=lambda pair: pair[1])
    return candidates[:max_chunks]
