import os
from functools import lru_cache
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseModel):
    TELEGRAM_BOT_TOKEN: str

    SAMBANOVA_API_KEY: str
    SAMBANOVA_BASE_URL: str = "https://api.sambanova.ai/v1"
    LLM_MODEL: str = "llama-3.1-70b-instruct"

    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    MEMORY_DB_PATH: str = "memory/dialogue.db"
    FAISS_INDEX_PATH: str = "vector_store/index.faiss"
    FAISS_META_PATH: str = "vector_store/meta.json"

    NOTION_TOKEN: str | None = None
    NOTION_DATABASE_ID: str | None = None

    BOT_ADMINS: str | None = None

    class Config:
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        TELEGRAM_BOT_TOKEN=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        SAMBANOVA_API_KEY=os.getenv("SAMBANOVA_API_KEY", ""),
        SAMBANOVA_BASE_URL=os.getenv("SAMBANOVA_BASE_URL", "https://api.sambanova.ai/v1"),
        LLM_MODEL=os.getenv("LLM_MODEL", "llama-3.1-70b-instruct"),
        EMBEDDING_MODEL=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        MEMORY_DB_PATH=os.getenv("MEMORY_DB_PATH", "memory/dialogue.db"),
        FAISS_INDEX_PATH=os.getenv("FAISS_INDEX_PATH", "vector_store/index.faiss"),
        FAISS_META_PATH=os.getenv("FAISS_META_PATH", "vector_store/meta.json"),
        NOTION_TOKEN=os.getenv("NOTION_TOKEN"),
        NOTION_DATABASE_ID=os.getenv("NOTION_DATABASE_ID"),
        BOT_ADMINS=os.getenv("BOT_ADMINS"),
    )
