import logging
from typing import List, Dict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.docstore.document import Document

logger = logging.getLogger(__name__)

SYSTEM_STRICT = (
    "You are a careful assistant. Answer only from the provided context. If the answer is not present, say you do not have enough information."
)
SYSTEM_HUMAN = (
    "You are a helpful assistant. Prefer clear, concise answers grounded in the provided context."
)

DRAFT_HINTS = {
    "book_outline": "Produce a structured book outline with parts, chapters and brief bullet points, grounded in the context.",
    "book_chapter": "Write a coherent chapter that cites facts back to the context. Keep it factual where appropriate.",
    "long_report": "Write a structured report with sections, a summary, key findings and citations.",
}

def build_prompt(context_blocks: List[str], question: str, tone: str, draft_mode: str | None) -> str:
    system = SYSTEM_STRICT if tone == "strict" else SYSTEM_HUMAN
    mode = DRAFT_HINTS.get(draft_mode, "Answer the question directly using the context.")
    context = "\n\n".join(context_blocks)
    prompt = (
        f"{system}\n\n"
        f"Task: {mode}\n\n"
        f"Context:\n{context}\n\n"
        f"User question or instruction:\n{question}\n\n"
        f"Rules: cite sources inline like [filename p.page]. If unsure, say so."
    )
    return prompt

def invoke_llm(llm: BaseChatModel, prompt: str) -> str:
    try:
        resp = llm.invoke(prompt)
        if hasattr(resp, "content"):
            return resp.content
        return str(resp)
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        raise
