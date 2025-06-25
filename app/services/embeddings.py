import numpy as np
from app.services.ml_models import load_embedding_model

# Simple in-memory cache (TODO: replace with Redis in production)
_cache = {}

def get_embedding(text: str) -> np.ndarray:
    """Get embedding for text with caching"""
    if text in _cache:
        return _cache[text]
    
    model = load_embedding_model()
    embedding = model.encode([text])[0]
    _cache[text] = embedding
    return embedding