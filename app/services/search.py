import logging
from app.services.embeddings import get_embedding
from app.utils.similarity import find_similar_items
from app.data.loader import load_psychology_data

logger = logging.getLogger(__name__)

async def semantic_search(query: str, top_k: int = 5):
    """Perform semantic search on psychology concepts"""
    try:
        # Get precomputed embeddings and texts
        embeddings, texts = load_psychology_data()

        if len(texts) == 0:
            logger.error("No texts found in dataset")
            return []
        
        logger.info(f"Searching with {len(texts)} concepts")

        # Generate query embedding
        query_embedding = get_embedding(query)

        # Find most similar items
        results = find_similar_items(
            query_embedding=query_embedding,
            embeddings=embeddings,
            texts=texts,
            top_k=top_k
        )

        return results
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        return []