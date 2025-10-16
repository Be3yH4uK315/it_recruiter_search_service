from pydantic import field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    """Класс для управления конфигурацией приложения."""
    ELASTICSEARCH_URL: str
    CANDIDATE_API_URL: str
    RABBITMQ_HOST: str
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASS: str = "guest"
    CANDIDATE_EXCHANGE_NAME: str
    CANDIDATE_ALIAS: str = "candidates"
    BATCH_SIZE: int = 500
    RRF_K: int = 60
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: str = "19530"
    MILVUS_INDEX_PARAMS: dict = {
        "metric_type": "IP",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    SENTENCE_MODEL_NAME: str = "paraphrase-multilingual-mpnet-base-v2"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()