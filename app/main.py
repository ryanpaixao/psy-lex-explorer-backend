from fastapi import FastAPI
from app.api import search, concepts, recommend
from app.core.config import settings

app = FastAPI(
    title="PsyLex API",
    description="Psychology Semeantic Search Engine",
    version="0.1.0"
)

# Register API routers
app.include_router(search.router, prefix="/api/search")
app.include_router(concepts.router, prefix="/api/concepts")
app.include_router(recommend.router, prefix="/api/recommend")

@app.on_event("startup")
async def startup_event():
    from app.services.ml_models import load_embedding_model
    # Preload ML models during startup
    load_embedding_model()