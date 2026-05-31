from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    database_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://rag_user:rag_password@localhost:5432/rag_assistant",
    )
    min_retrieval_similarity: float = float(os.getenv("MIN_RETRIEVAL_SIMILARITY", "0.18"))
    top_k: int = int(os.getenv("TOP_K", "5"))
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


settings = Settings()
