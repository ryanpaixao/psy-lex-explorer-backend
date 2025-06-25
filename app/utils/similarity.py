import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_items(
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        texts: list[str],
        top_k: int = 5
) -> list[dict]:
    """Find top_k similar items using cosine similarity"""
    # Calclulate cosine similarity
    similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]

    # Get top indices
    top_indices = similarities.argsort()[::-1][:top_k]

    # Format results
    return [
        {
            "text": texts[i],
            "score": float(similarities[i]),
            "index": int(i)
        }
        for i in top_indices
    ]