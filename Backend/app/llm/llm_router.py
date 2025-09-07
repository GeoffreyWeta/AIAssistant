import logging
from typing import Optional

from app.Config.config import settings

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

logger = logging.getLogger(__name__)

# Map UI names to concrete chat models and tokenizer ids
MODEL_MAP = {
    "openai": {"chat_model": "gpt-4o-mini", "tokenizer": "gpt-4o-mini"},
    "gpt": {"chat_model": "gpt-4o-mini", "tokenizer": "gpt-4o-mini"},
    "groq": {"chat_model": "llama-3.1-8b-instant", "tokenizer": "llama-3"},
    "llama3": {"chat_model": "llama-3.1-8b-instant", "tokenizer": "llama-3"},
    "huggingface": {"chat_model": "hf-bart", "tokenizer": "cl100k_base"},
    "hf-bart": {"chat_model": "hf-bart", "tokenizer": "cl100k_base"},
}

def get_llm(model_name: Optional[str]):
    """
    Returns a LangChain compatible chat model.
    Takes a UI hint, resolves to a concrete model.
    """
    key = (model_name or settings.DEFAULT_LLM).lower()
    cfg = MODEL_MAP.get(key)
    if not cfg:
        raise ValueError(f"Unsupported LLM model: {model_name}")

    chat_model = cfg["chat_model"]

    if chat_model.startswith("gpt-"):
        if not settings.OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY in environment")
        logger.info(f"Using OpenAI chat model: {chat_model}")
        # ChatOpenAI reads OPENAI_API_KEY from env
        return ChatOpenAI(model=chat_model, temperature=0)

    if chat_model.startswith("llama"):
        if not settings.GROQ_API_KEY:
            raise ValueError("Missing GROQ_API_KEY in environment")
        logger.info(f"Using Groq chat model: {chat_model}")
        # ChatGroq reads GROQ_API_KEY from env
        return ChatGroq(model_name=chat_model, temperature=0)

    if chat_model == "hf-bart":
        logger.info("Using HuggingFace local summarization model for lightweight tasks")
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        return HuggingFacePipeline(pipeline=summarizer)

    raise ValueError(f"Unsupported chat model: {chat_model}")

def get_tokenizer_name(model_name: Optional[str]) -> str:
    key = (model_name or settings.DEFAULT_LLM).lower()
    cfg = MODEL_MAP.get(key) or MODEL_MAP["openai"]
    return cfg["tokenizer"]
