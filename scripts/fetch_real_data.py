import sys
import os
from pathlib import Path

# Add parent directory to path to enable app imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
from app.data.preprocess import clean_text
import json
import requests
import time

def fetch_psychology_glossary():
    """Fetch psychology glossary from GitHub"""
    try:
        # Original url - url = "https://raw.githubusercontent.com/dylan-profiler/glossary-of-psychology-terms/main/glossary.json"
        # Alt url - url = "https://raw.githubusercontent.com/psychology-lexicon/psychology-lexicon/main/glossary.json"
        url = "https://raw.githubusercontent.com/eddienko/psychology_terms_eng/master/psychology_terms.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Inspect the response content
        print(f"Response status: {response.status_code}")
        print(f"First 100 characters: {response.text[:100]}")

        # Attempt to parse JSON
        glossary = response.json()
        print(f"Successfully parsed {len(glossary)} glossary terms")

        return [
            f"{term}: {definition}"
            for term, definition in glossary.items()
        ]
    except Exception as e:
        print(f"Error fetching glossary: {str(e)}")
        return []

def fetch_openalex_psychology_abstracts(max_results=500):
    """Fetch psychology paper abstracts from OpenAlex"""
    try:
        base_url = "https://api.openalex.org/works"
        params = {
            "filter": "concepts.id:C71924100", # Psychology concept ID
            "per-page": 200,
            "select": "id,title,abstract,publication_year"
        }

        papers = []
        page = 1
        while len(papers) < max_results:
            params["page"] = page
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                print(f"Error fetching page {page}: {response.status_code}")
                break

            data = response.json()
            results = data.get("results", [])
            if not results:
                break

            for work in results:
                if work.get("abstract"):
                    papers.append({
                        "text": f"{work['title']}, {clean_text(work['abstract'])}",
                        "source": "OpenAlex",
                        "year": work.get("publication_year")
                    })

            print(f"Fetched {len(papers)} papers so far...")
            page += 1
            time.sleep(0.3) # Respect API rate limits

        return [paper["text"] for paper in papers]
    except Exception as e:
        print(f"Error fetching OpenAlex data: {str(e)}")
        return []

def fetch_cognitive_atlas_concepts():
    """Fetch Cognitive Atlas concepts"""
    try:
        url = "https://www.cognitiveatlas.org/api/v-alpha/concepts"
        response = requests.get(url)
        concepts = response.json()

        return [
            f"{concept['name']}: {concept['definition_text']}"
            for concept in concepts
        ]
    except Exception as e:
        print(f"Error fetching Cognitive Atlas: {str(e)}")
        return []

def main():
    # Fetch data from multiple sources
    print("Fetching psychology glossary...")
    glossary_terms = fetch_psychology_glossary()

    print("Fetching Cognitive Atlas concepts...")
    cognitive_concepts = fetch_cognitive_atlas_concepts()

    print("Fetching OpenAlex psychology abstracts...")
    abstracts = fetch_openalex_psychology_abstracts(max_results=300)

    # Combine all data
    all_concepts = glossary_terms + cognitive_concepts + abstracts
    print(f"Total concepts: {len(all_concepts)}")

    # Save raw data
    raw_path = settings.DATA_PATH / "raw_psychology_data.json"
    with open(raw_path, "w") as f:
        json.dump(all_concepts, f, indent=2)
    print(f"Saved raw data to {raw_path}")

    # Process and prepare for embedding
    processed_data = [{"text": text} for text in all_concepts]

    # Save to main dataset
    output_path = settings.DATA_PATH / "psychology_concepts.json"
    with open(output_path, "w") as f:
        json.dump(processed_data, f, indent=2)
    print("Now run: python -m scripts.precompute_embeddings")

if __name__ == "__main__":
    main()