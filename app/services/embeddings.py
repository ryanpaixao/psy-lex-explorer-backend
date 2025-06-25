import numpy as np
from sentence_transformers import SentenceTransformer
from app.utils.cache import cache

_model = None

def load_embedding_model():
    """Load and cache the embedding model"""
    global _model
    if _model is None:
        _model = SentenceTransformer('all-MiniLM-L6-v2')
    return _model

@cache(ttl=3600) # Cache embeddings for 1 hour
def get_embedding(text: str) -> np.ndarray:
    """Get embedding for a text with caching"""
    model = load_embedding_model()
    return model.encode([text])[0]