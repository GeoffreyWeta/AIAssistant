import os
import json
import uuid
import time
import logging
import re
import difflib
from urllib.parse import urlparse
from typing import Optional, List, Dict, Any

import tiktoken
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.Config.config import settings
from app.documents.document_loader import load_and_vectorize
from app.vectorstore.vector_store import multi_similarity_search_with_score
from app.llm.llm_router import get_llm, get_tokenizer_name
from app.chains.rag_chain import build_prompt, invoke_llm
from app.utils.prompts import build_context_blocks
from app.services.reporting import save_pdf_report

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI
app = FastAPI(title="RAG FastAPI App", version="2.3")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # set explicit origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static for reports download
os.makedirs(settings.REPORTS_DIR, exist_ok=True)
app.mount("/reports", StaticFiles(directory=settings.REPORTS_DIR), name="reports")

# -------------------------
# Helpers
# -------------------------
def count_tokens(text: str, tokenizer_name: str) -> int:
    try:
        enc = tiktoken.encoding_for_model(tokenizer_name)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def budget_docs(
    docs_with_scores: List[tuple],
    tokenizer_name: str,
    max_context_tokens: int = 6000,
    reserve_for_prompt: int = 1000
) -> List[tuple]:
    enc = tiktoken.get_encoding("cl100k_base")
    current = 0
    kept: List[tuple] = []
    for doc, score in docs_with_scores:
        t = enc.encode(doc.page_content)
        if current + len(t) > max_context_tokens - reserve_for_prompt:
            break
        kept.append((doc, score))
        current += len(t)
    return kept

def find_doc_ids_for_filenames(filenames: List[str]) -> List[str]:
    """
    Scan vector_db folders, read meta.json and map selected filenames to doc_ids.
    """
    doc_ids: List[str] = []
    root = settings.VECTOR_DB_DIR
    if not os.path.exists(root):
        return []
    for folder in os.listdir(root):
        meta_file = os.path.join(root, folder, "meta.json")
        if not os.path.exists(meta_file):
            continue
        with open(meta_file) as f:
            m = json.load(f)
            if m.get("filename") in filenames:
                doc_ids.append(m.get("id") or folder)
    return doc_ids

def validate_upload(filename: str, size_bytes: int):
    allowed = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type {ext}")
    mb = size_bytes / (1024 * 1024)
    if mb > settings.MAX_UPLOAD_MB:
        raise HTTPException(status_code=400, detail=f"File too large, limit is {settings.MAX_UPLOAD_MB} MB")

# -------------------------
# Web enrichment, DuckDuckGo
# -------------------------
def maybe_web_enrich(
    query: str,
    k: int = 5,
    region: str = "us-en",
    safesearch: str = "moderate",  # off, moderate, strict
    timeout: int = 6,
) -> list[dict]:
    """
    Text web search using duckduckgo_search.
    Returns: [{ title, url, snippet }]
    """
    if not query or not query.strip():
        return []
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except Exception:
        logger.warning("duckduckgo_search is not installed, skipping web enrichment")
        return []
    try:
        with DDGS(timeout=timeout) as ddgs:
            results = ddgs.text(
                query,
                region=region,
                safesearch=safesearch,
                max_results=k,
            ) or []
        out: list[dict] = []
        for r in results:
            title = r.get("title") or ""
            url = r.get("href") or r.get("url") or ""
            snippet = r.get("body") or r.get("snippet") or ""
            if title and url:
                out.append({"title": title, "url": url, "snippet": snippet})
        return out
    except Exception as e:
        logger.exception("maybe_web_enrich failed: %s", e)
        return []

# Focus helpers
def _extract_focus_from_query(query: str) -> str:
    q = query.strip()

    m = re.search(r'"([^"]+)"', q)
    if m:
        return m.group(1).strip()

    leadins = [
        r"show me( images| photos)? of",
        r"images? of",
        r"photos? of",
        r"pictures? of",
        r"what is",
        r"who is",
        r"tell me about",
        r"explain",
        r"how does",
        r"how do",
        r"how to",
    ]
    q = re.sub(r"^(?:{:s})\s+".format("|".join(leadins)), "", q, flags=re.IGNORECASE).strip()

    cleaned = re.sub(r"[^a-z0-9\s\-]", " ", q.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    stop = {
        "a","an","the","of","and","or","to","for","in","on","at","by","with","about","from",
        "show","me","images","image","photos","photo","pictures","picture","diagram","chart",
        "explain","what","who","how","does","do","is","are","was","were","be","being","been",
        "please","give","find","latest",
    }
    tokens = [t for t in cleaned.split() if t not in stop and len(t) > 1]
    return " ".join(tokens[:8]) if tokens else query.strip()

def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _host(u: str) -> str:
    try:
        return urlparse(u).hostname or ""
    except Exception:
        return ""

def derive_image_query_from_answer(answer: str, original_q: str, max_chars: int = 120) -> str:
    """
    Very light heuristic:
    1) look for quoted phrases in the answer
    2) else look for capitalized multi word spans
    3) else fall back to focus from the original query
    """
    # 1) quoted phrase
    m = re.search(r'"([^"]{3,120})"', answer)
    if m:
        return m.group(1).strip()[:max_chars]

    # 2) capitalized spans
    caps = re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", answer)
    if caps:
        cand = sorted(set(caps), key=len, reverse=True)[0]
        return cand[:max_chars]

    # 3) fallback
    return _extract_focus_from_query(original_q)[:max_chars]

def maybe_web_images(
    query: str,
    k: int = 6,
    safesearch: str = "moderate",   # off, moderate, strict
    size: str = "Medium",           # Small, Medium, Large, Wallpaper
    timeout: int = 6,
) -> list[dict]:
    """
    Image search using duckduckgo_search, biased toward the query focus.
    Returns: [{ title, url, image_url, thumbnail, width, height, score }]
    """
    if not query or not query.strip():
        return []
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except Exception:
        logger.warning("duckduckgo_search is not installed, skipping web image search")
        return []

    focus_text = _extract_focus_from_query(query)

    # simple intent tweak
    if re.search(r"\b(architecture|pipeline|schematic|flow|diagram|design|topology|uml|erd)\b", query, re.I):
        q = f"{focus_text} diagram"
    elif re.search(r"\b(who is|portrait|headshot|city|bridge|mountain|building|landmark)\b", query, re.I):
        q = f"{focus_text} photo"
    else:
        q = focus_text

    try:
        with DDGS(timeout=timeout) as ddgs:
            results = ddgs.images(
                q,
                safesearch=safesearch,
                size=size,
                max_results=max(k * 2, 10),
            ) or []
        gathered: list[dict] = []
        seen = set()
        for r in results:
            image_url = r.get("image") or r.get("thumbnail") or ""
            if not image_url:
                continue
            key = image_url.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            title = r.get("title") or ""
            src_page = r.get("source") or r.get("url") or ""
            thumb = r.get("thumbnail") or ""
            width = r.get("width")
            height = r.get("height")
            try:
                w = int(width) if width is not None else None
                h = int(height) if height is not None else None
            except Exception:
                w, h = None, None
            if w is not None and h is not None and (w < 256 or h < 256):
                continue
            gathered.append({
                "title": title,
                "url": src_page,
                "image_url": image_url,
                "thumbnail": thumb,
                "width": w,
                "height": h,
            })
    except Exception as e:
        logger.exception("maybe_web_images failed: %s", e)
        return []

    if not gathered:
        return []

    def score_item(it: dict) -> float:
        title = it.get("title") or ""
        host = _host(it.get("url") or "")
        sim = max(_similarity(title, focus_text), _similarity(host.replace("www.", ""), focus_text))
        w = it.get("width") or 0
        h = it.get("height") or 0
        res = min((w * h) / (1920 * 1080), 1.0) if w and h else 0.3
        return 0.7 * sim + 0.3 * res

    gathered.sort(key=score_item, reverse=True)
    topk = gathered[:k]
    for it in topk:
        it["score"] = round(score_item(it), 4)
    return topk

# -------------------------
# Models
# -------------------------
class ChatRequest(BaseModel):
    q: str
    documents: List[str] = Field(default_factory=list)  # filenames
    llm_name: Optional[str] = None
    web_enrich: bool = False
    return_mode: str = Field(default="chat")  # chat or pdf
    tone: str = Field(default=settings.DEFAULT_TONE)  # human or strict
    draft_mode: Optional[str] = Field(default=None)  # None, book_outline, book_chapter, long_report
    max_chunks: int = 8
    rerank: bool = False
    citations: bool = True
    # new controls for image search timing
    image_source: str = Field(default="auto")  # "query", "answer", "auto"
    image_k: int = Field(default=6)

# -------------------------
# Upload
# -------------------------
@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    os.makedirs(settings.TEMP_UPLOADS_DIR, exist_ok=True)
    os.makedirs(settings.VECTOR_DB_DIR, exist_ok=True)

    results = []
    for file in files:
        content = await file.read()
        validate_upload(file.filename, len(content))

        # Save raw upload
        file_path = os.path.join(settings.TEMP_UPLOADS_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(content)
        logger.info(f"Saved upload {file.filename}")

        # Create doc folder
        doc_id = str(uuid.uuid4())
        vector_path = os.path.join(settings.VECTOR_DB_DIR, doc_id)
        os.makedirs(vector_path, exist_ok=True)

        # Save meta
        meta = {"id": doc_id, "filename": file.filename}
        with open(os.path.join(vector_path, "meta.json"), "w") as mf:
            json.dump(meta, mf)

        # Vectorize
        try:
            load_and_vectorize(
                file_path=file_path,
                doc_id=doc_id,
                filename=file.filename,
                persist_directory=vector_path,
                summarize=False
            )
        except Exception as e:
            logger.error(f"Vectorization failed for {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Vectorization failed for {file.filename}")

        results.append(meta)

    return {"status": f"{len(results)} file(s) uploaded and indexed", "documents": results}

# -------------------------
# Documents list and delete
# -------------------------
@app.get("/documents")
async def list_documents():
    root = settings.VECTOR_DB_DIR
    docs = []
    if os.path.exists(root):
        for folder in os.listdir(root):
            meta_file = os.path.join(root, folder, "meta.json")
        # corrected indentation
            if os.path.exists(meta_file):
                with open(meta_file) as f:
                    docs.append(json.load(f))
    return {"documents": docs}

@app.delete("/documents/{file_name}")
async def delete_document(file_name: str):
    try:
        raw_path = os.path.join(settings.TEMP_UPLOADS_DIR, file_name)
        if os.path.exists(raw_path):
            os.remove(raw_path)
        vector_db_root = settings.VECTOR_DB_DIR
        deleted = False
        for folder in os.listdir(vector_db_root):
            meta_file = os.path.join(vector_db_root, folder, "meta.json")
            if not os.path.exists(meta_file):
                continue
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get("filename") == file_name:
                import shutil
                shutil.rmtree(os.path.join(vector_db_root, folder), ignore_errors=True)
                deleted = True
                break
        if not deleted:
            logger.warning(f"No vector DB found for {file_name}")
        return {"status": f"Document '{file_name}' deleted"}
    except Exception:
        raise HTTPException(status_code=500, detail=f"Failed to delete {file_name}")

# -------------------------
# Search and generate
# -------------------------
@app.post("/search")
async def chat(request: ChatRequest):
    t0 = time.time()
    if not request.documents:
        raise HTTPException(status_code=400, detail="No document selected")

    # Resolve filenames to doc_ids
    doc_ids = find_doc_ids_for_filenames(request.documents)
    if not doc_ids:
        raise HTTPException(status_code=404, detail="No matching documents found")

    # Build retrieval set
    hits = multi_similarity_search_with_score(
        doc_ids=doc_ids,
        query=request.q,
        max_chunks=request.max_chunks
    )

    # Token budget
    tokenizer_name = get_tokenizer_name(request.llm_name)
    hits = budget_docs(hits, tokenizer_name)

    # Build context and citations
    context_blocks, citations = build_context_blocks(hits)

    # Optional web enrichment, text only, used to inform the answer
    web_sources = maybe_web_enrich(request.q) if request.web_enrich else []
    if web_sources:
        ext_block = "\n".join([f"[web] {s.get('snippet','')} ({s.get('url','')})" for s in web_sources])
        context_blocks.append(ext_block)

    # Build final prompt and answer
    prompt = build_prompt(
        context_blocks=context_blocks,
        question=request.q,
        tone=request.tone,
        draft_mode=request.draft_mode,
    )

    llm = get_llm(request.llm_name)
    answer_text = invoke_llm(llm, prompt)

    # Image enrichment after answer if requested
    web_images: list[dict] = []
    if request.web_enrich:
        if request.image_source == "query":
            web_images = maybe_web_images(request.q, k=request.image_k)
        elif request.image_source == "answer":
            derived = derive_image_query_from_answer(answer_text, request.q)
            web_images = maybe_web_images(derived, k=request.image_k)
        else:  # auto
            derived = derive_image_query_from_answer(answer_text, request.q)
            web_images = maybe_web_images(derived, k=request.image_k) or maybe_web_images(request.q, k=request.image_k)

    latency_ms = int((time.time() - t0) * 1000)

    # Return as PDF if requested
    if request.return_mode == "pdf":
        report = save_pdf_report(
            answer=answer_text,
            citations=citations,
            question=request.q,
            web_sources=web_sources,
        )
        url = f"/reports/{report['report_id']}.pdf"
        return {
            "mode": "pdf",
            "report_id": report["report_id"],
            "url": url,
            "citations": citations,
            "web_sources": web_sources,
            "web_images": web_images,
            "latency_ms": latency_ms,
        }

    # Default chat mode
    return {
        "mode": "chat",
        "answer": answer_text,
        "citations": citations if request.citations else [],
        "web_sources": web_sources,
        "web_images": web_images,
        "latency_ms": latency_ms,
    }

# -------------------------
# Health and stats
# -------------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/stats")
def stats():
    total_docs = 0
    total_chunks = 0
    root = settings.VECTOR_DB_DIR
    if os.path.exists(root):
        for folder in os.listdir(root):
            store_path = os.path.join(root, folder)
            if not os.path.isdir(store_path):
                continue
            total_docs += 1
            for name in os.listdir(store_path):
                if name.endswith(".sqlite3"):
                    total_chunks += 1  # rough proxy
    return {"total_docs": total_docs, "approx_collections": total_docs, "approx_indexes": total_chunks}
