from pydantic_settings import BaseSettings
from pydantic import ConfigDict, field_validator

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = ""
    gemini_api_keys: str = ""
    groq_api_key: str = ""
    embedding_provider: str = "gemini"
    generation_provider: str = "gemini"
    judge_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-2"
    embedding_dimension: int = 768
    generation_model: str = "gemini-2.5-flash"
    groq_model: str = "llama-3.3-70b-versatile"
    judge_model: str = ""
    gemini_max_retries: int = 2
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k: int = 5
    max_retrieval_distance: float | None = 0.90
    data_root: str = "data"
    db_path: str = "db/lancedb"
    cache_path: str = "cache/diskcache"
    reports_path: str = "reports"

    @field_validator("max_retrieval_distance", mode="before")
    @classmethod
    def empty_distance_to_none(cls, value):
        if value == "":
            return None
        return value

settings = Settings()
