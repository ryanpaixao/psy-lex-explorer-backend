import json
import numpy as np
import logging
from pathlib import Path
from app.core.config import settings

logger = logging.getLogger(__name__)

def load_psychology_data():
    """Load psychology concepts and precomuted embeddings"""
    data_path = settings.DATA_PATH / "psychology_concepts.json"
    logger.info(f"Loading data from: {data_path}")


    if not data_path.exists():
        # Create sample data if none exists
        logger.warning("Data file not found. Creating sample data.")
        return create_sample_data()

    try:
        with open(data_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} items")

        # Validate data structure
        if not isinstance(data, list) or len(data) == 0:
            logger.error("Invalid data format. Creating sample data.")
            return create_sample_data()

        # Convert to numpy arrays
        texts = [item["text"] for item in data]
        embeddings = np.array([item["embedding"] for item in data])

        return embeddings, texts
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return create_sample_data()

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
    try:
        with open(settings.DATA_PATH / "psychology_concepts.json", "w") as f:
            json.dump(sample_data, f, indent=2)
        logger.info(f"Sample data saved to {settings.DATA_PATH}/psychology_concepts.json")
    except Exception as e:
        logger.error(f"Failed to save sample data: {str(e)}")

    texts = [item["texts"] for item in sample_data]
    embeddings = np.array([item["embedding"] for item in sample_data])

    return embeddings, texts