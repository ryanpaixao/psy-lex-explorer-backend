import json
import requests
from pathlib import Path
from app.core.config import settings
from app.services.embeddings import get_embedding

def fetch_psychology_concepts():
    """Fetch sample psychology concepts (TODO: replace with real data source)"""
    return [
        "Cognitive dissonance: The mental discomfort experienced when holding conflicting beliefs",
        "Classical conditioning: Learning through association (Pavlov's dogs)",
        "Operant conditioning: Learning through consequences (Skinner's box)",
        "Theory of mind: Understanding others have different mental states",
        "Confirmation bias: Favoring information that confirms existing beliefs",
        "Neuroplasticity: The brain's ability to reorganize neural pathways",
        "Working memory: Short-term cognitive storage and manipulation",
        "Flow state: Complete immersion in an activity"
    ]

def main():
    concepts = fetch_psychology_concepts()

    processed_data = []
    for concept in concepts:
        embedding = get_embedding(concept).tolist()
        processed_data.append({
            "text": concept,
            "embedding": embedding
        })

    # Save to file
    settings.DATA_PATH.mkdir(parents=True, exist_ok=True)
    output_path = settings.DATA_PATH / "psychology_concepts.json"

    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=2)

    print(f"Saved {len(processed_data)} psychology concepts to {output_path}")

if __name__ == "__main__":
    main()