import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from app.core.config import settings
from app.data.preprocess import clean_text

def main():
    # Load model
    model = SentenceTransformer(settings.EMBEDDING_MODEL)

    # Load the data
    data_path = settings.DATA_PATH / "psychology_concepts.json"
    with open(data_path, "r") as f:
        data = json.load(f)

    print(f"Loaded {len(data)} psychology concepts")

    # Clean text and compute embeddings
    for item in tqdm(data, desc="Computing embeddings"):
        item["text"] = clean_text(item["text"]).tolist()
        embedding = model.encode([item["text"]])[0]
        item["embedding"] = embedding.tolist()

    # Save with embeddings
    with open(data_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved embeddings for {len(data)} concepts to {data_path}")

if __name__ == "__main__":
    main()