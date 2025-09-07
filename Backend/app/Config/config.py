import os
import logging
from dotenv import load_dotenv

# Load .env file into environment
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Settings:

    GOOGLE_CSE_KEY: str = os.getenv("GOOGLE_CSE_KEY", "")
    GOOGLE_CSE_CX: str = os.getenv("GOOGLE_CSE_CX", "")
    GOOGLE_SAFE: str = os.getenv("GOOGLE_SAFE", "active")  # active or off
    GOOGLE_GL: str = os.getenv("GOOGLE_GL", "us")          # country code, example ng, us, gb


    # LLM selection defaults
    DEFAULT_LLM: str = os.getenv("DEFAULT_LLM", "openai")
    OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY: str | None = os.getenv("GROQ_API_KEY")

    # App defaults
    DEFAULT_TONE: str = os.getenv("DEFAULT_TONE", "human")  # human or strict
    VECTOR_DB_DIR: str = os.getenv("VECTOR_DB_DIR", "vector_db")
    TEMP_UPLOADS_DIR: str = os.getenv("TEMP_UPLOADS_DIR", "temp_uploads")
    REPORTS_DIR: str = os.getenv("REPORTS_DIR", "reports")
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "25"))

settings = Settings()
logger.info("Settings loaded from environment")

