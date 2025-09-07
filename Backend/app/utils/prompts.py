from typing import List, Tuple
from langchain.docstore.document import Document

def build_context_blocks(docs_with_scores: List[Tuple[Document, float]]) -> tuple[list[str], list[dict]]:
    blocks: list[str] = []
    citations: list[dict] = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        filename = doc.metadata.get("filename", "document")
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content.strip().replace("\n", " ")
        tag = f"[{filename} p.{page}]"
        blocks.append(f"{tag} {snippet}")
        citations.append({
            "index": i,
            "filename": filename,
            "page": page,
            "snippet": snippet[:400],
            "doc_id": doc.metadata.get("doc_id"),
            "score": float(score),
        })
    return blocks, citations
