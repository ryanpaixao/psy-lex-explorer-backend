import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

def find_similar_items(
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        texts: list[str],
        top_k: int = 5
) -> list[dict]:
    """Find top_k similar items using cosine similarity"""
    try:
        # Validate inputs
        if embeddings.shape[0] == 0:
            logger.error("No embeddings provided")
            return []
        
        if len(texts) != embeddings.shape[0]:
            logger.error("Mismatch between texts and embeddings count")
            return []

        # Calclulate cosine similarity
        similarities = cosine_similarity([query_embedding], embeddings)[0]

        # Get top indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Format results
        return [
            {
                "text": texts[i],
                "score": float(similarities[i]),
                "index": int(i)
            }
            for i in top_indices
        ]
    except Exception as e:
        logger.error(f"Similarity calculation error: {str(e)}")
        return []