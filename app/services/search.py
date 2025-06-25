from app.services.embeddings import get_embedding
from app.utils.similarity import find_similar_items
from app.data.loader import get_psychology_data

async def semantic_search(query: str, top_k: int = 5):
    """Perform semantic search on psychology concepts"""
    # Get precomputed embeddings and texts
    embeddings, texts = get_psychology_data()

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