"""
Configuration centrale du RAG contech (construction cost estimating).
Modifie ces valeurs selon tes besoins.
"""

import os
from pathlib import Path

# --- Dossiers ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"          # Mets ici tes PDFs / .txt / .md
DB_DIR = BASE_DIR / "chroma_db"       # Base vectorielle locale (auto-créée)

# --- Chunking ---
CHUNK_SIZE = 800        # caractères par chunk
CHUNK_OVERLAP = 150     # chevauchement entre chunks (garde le contexte)

# --- Embeddings (modèle local, gratuit, tourne sans clé API) ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval ---
TOP_K = 5  # nombre de chunks les plus pertinents à récupérer par question
DEFAULT_FILTERS = {
    "source": None,
    "tool": None,
    "category": None,
    "cost": None,
}

# --- Génération (Claude) ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-6"

# --- Collection ChromaDB ---
COLLECTION_NAME = "contech_cost_estimating"

SYSTEM_PROMPT = """Tu es un assistant spécialisé en construction technology (contech),
en particulier les outils d'estimation de coûts de construction (Buildertrend, Procore,
PlanSwift, STACK, ProEst, RSMeans, etc.).

Réponds UNIQUEMENT à partir du contexte fourni ci-dessous, extrait des documents indexés.
Si le contexte ne contient pas la réponse, dis-le clairement au lieu d'inventer.
Cite le nom du document source quand c'est pertinent.
Réponds de façon claire et structurée, utile pour préparer des comparatifs ou des vidéos."""
