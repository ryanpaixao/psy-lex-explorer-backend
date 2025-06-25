import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_ENV: str = os.getenv("APP_ENV", "development")
    DATA_PATH: Path(os.getenv("DATA_PATH", "./app/data/datasets/"))
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", 3600))

settings = Settings()