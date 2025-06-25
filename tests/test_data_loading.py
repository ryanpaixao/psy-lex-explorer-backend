import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import logging
from app.data.loader import load_psychology_data

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    embeddings, texts = load_psychology_data()
    print(f"Loaded {len(texts)} texts")
    print(f"Sample text: {texts[0] if texts else 'No texts'}")
    print(f"Embeddings shape: {embeddings.shape if embeddings is not None else 'No embeddings'}")