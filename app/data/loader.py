import json
import numpy as np
from pathlib import Path
from app.core.config import DATA_DIR

def load_psychology_data():
    """Load psychology concepts and precomuted embeddings"""
    data_path = Path(DATA_DIR) / "psychology_concetps.json"

    with open(data_path, "r") as f:
        data = json.load(f)

    # Convert to numpy arrays
    texts = [item["text"] for item in data]
    embeddings = np.array([item["embeddings"] for item in data])

    return embeddings, texts