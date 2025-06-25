import logging
from fastapi import FastAPI
from app.api import search
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PsyLex API",
    description="Psychology Semeantic Search Engine",
    version="0.1.0"
)

# Register API routers
app.include_router(search.router, prefix="/api/search")
# app.include_router(concepts.router, prefix="/api/concepts")
# app.include_router(recommend.router, prefix="/api/recommend")

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")

    # Preload ML models during startup
    from app.services.ml_models import load_embedding_model
    load_embedding_model()

    # Test data loading
    from app.data.loader import load_psychology_data
    embeddings, texts = load_psychology_data()
    logger.info(f"Loaded {len(texts)} psychologoy concepts")

    logger.info("Application startup complete")