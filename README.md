# Tech Stack #
**Frontend**: React 19 (with new hooks), TypeScript, Tailwind CSS

**Backend**: Python (Flask/FastAPI), NLP libraries (spaCy, scikit-learn)

**ML**: Sentence Transformers, BERTopic, FAISS (similarity search)

**Data**: Psychology textbook corpus, research abstracts, glossary databases


# Core Idea #
Build an intelligent search interface that goes beyond keyword matching to:
1) Understand psychological concepts contextually
2) Map relationships between theories/concepts
3) Recommend semantically related research
4) Visualize knowledge networks


# Key Features & Implementation #
1) Semantic Search Engine
  - ML Component: Sentence-BERT embeddings + FAISS index
  - Backend(Python)
  - Frontend(React/TS)
    - Search component with debounced input
    - Results card with confidence score visualization
2) Concept Relationship Mapper
  - ML Component: BERTopic for topic modeling
  - Workflow:
    1) Extract key concepts from search results
    2) Generate topic clusters
    3) Visualize with force-directed graph
  - React Visulaization: Use react-force-graph or D3.js
3) Literature Recommender
  - ML Component: Cosine similarity on TF-IDF vectors
  - Feature: "Find similar papers" button on each result
4) Cognitive Bias Detector(Bonus)
  - Analyze search patterns to suggest:
    - Confirmation bias in queries
    - Overlooked perspectives
  - Use rule-based NLP with spaCy patterns


# Psychology Data Sources #
1) Open Access Journals:
  - APA PsycArticles
  - PubMed Central (psychology subset)
2) Textbook Corpus:
  - Create structured glossary from psychology textbooks
3) Theory Databases:
  - Compile key theories(e.g., Maslow's hierarchy, CBT principles)
4) Sample Dataset Structure:
  ```
  {
    "concept": "Cognitive Dissonance",
    "theory": "Festinger (1957)",
    "definition": "The mental discomfort...",
    "related": ["Confirmation Bias", "Self-Perception Theory"],
    "studies": ["10.1037/h0041598", "10.1177/0146167205280909"]
  }
  ```


# Frontend Components (React 19 + TypeScript) #
1) SemanticSearch.tsx - Main search interface
2) KnowledgeGraph.tsx - 3D force-directed graph
3) TheoryCard.tsx - Animated concept cards
4) BiasDetector.tsx - Query analysis panel
5) RecommendationEngine.tsx - Carousel of related papers


# Backend Services (Python) #
1) /api/search - Semantic search endpoint
2) /api/visualize - Topic relationship data
3) /api/recommend - Similar papers suggestion
4) /api/analyze - Bias detection endpoint


# Machine Learning Pipeline #
graph LR
A[Raw Psychology Texts] --> B(Data Preprocessing)
B --> C[Embedding Generation]
C --> D[FAISS Indexing]
D --> E[Search Service]
B --> F[Topic Modeling]
F --> G[Knowledge Graph]
E --> H[React Interface]