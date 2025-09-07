import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

def get_embedding_model():
    """
    Loads and returns the HuggingFace embedding model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"Loading embedding model: {model_name}")
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}
    )
