import json
import numpy as np
from app.core.config import settings

def load_psychology_data():
    """Load psychology concepts and precomuted embeddings"""
    data_path = settings.DATA_PATH / "psychology_concetps.json"

    if not data_path.exists():
        # Create sample data if none exists
        return create_sample_data()

    with open(data_path, "r") as f:
        data = json.load(f)

    # Convert to numpy arrays
    texts = [item["text"] for item in data]
    embeddings = np.array([item["embedding"] for item in data])

    return embeddings, texts

def create_sample_data():
    """Create sample psychology data"""
    sample_data = [
        {"text": "Cognitive dissonance theory proposes that people seek consistency", "embedding": None},
        {"text": "Behaviorism focuses on observable behaviors and conditioning", "embedding": None},
        {"text": "Maslow's hierarchy of needs describes human motivation", "embedding": None},
        {"text": "Neuroplasticity refers to the brain's ability to reorganize itself", "embedding": None},
        {"text": "Social learning theory emphasizes observation and modeling", "embedding": None}
    ]

    # Generate embeddings for sample data
    from app.services.embeddings import get_embedding
    for item in sample_data:
        item["embedding"] = get_embedding(item["text"]).tolist()

    # Save sample data
    with open(settings.DATA_PATH / "psychology_concepts.json", "w") as f:
        json.dump(sample_data, f, indent=2)

    texts = [item["texts"] for item in sample_data]
    embeddings = np.array([item["embedding"] for item in sample_data])

    return embeddings, texts