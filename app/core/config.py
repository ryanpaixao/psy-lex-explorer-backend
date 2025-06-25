import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    def __init__(self):
        self.APP_ENV = os.getenv("APP_ENV", "development")
        self.DATA_PATH = Path(os.getenv("DATA_PATH", "./app/data/datasets/"))
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        self.CACHE_TTL = int(os.getenv("CACHE_TTL", 3600))

        # Create directories if they don't exist
        self.DATA_PATH.mkdir(parents=True, exist_ok=True)

settings = Settings()