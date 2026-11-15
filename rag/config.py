"""Settings, read from the environment (.env is loaded if present)."""
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional
    pass


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


LLM_BASE_URL = _get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = _get("LLM_API_KEY", "")
LLM_MODEL = _get("LLM_MODEL", "openai/gpt-4o-mini")

EMBED_MODEL = _get("EMBED_MODEL", "all-MiniLM-L6-v2")

CHUNK_SIZE = int(_get("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "50"))
TOP_K = int(_get("TOP_K", "5"))
MIN_SCORE = float(_get("MIN_SCORE", "0.25"))

INDEX_DIR = _get("INDEX_DIR", ".chroma")
COLLECTION = _get("COLLECTION", "docs")

IDK = "I don't know"
