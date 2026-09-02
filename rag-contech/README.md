# RAG Contech — Construction Cost Estimating

Prototype de RAG (Retrieval-Augmented Generation) spécialisé dans les outils
d'estimation de coûts de construction (Buildertrend, Procore, PlanSwift,
STACK, ProEst, RSMeans, etc.), pensé pour préparer des comparatifs et vidéos.

## Comment ça marche

```
data/ (tes PDFs/notes)
   │
   ▼
ingest.py  →  découpe en chunks → embeddings locaux → ChromaDB (chroma_db/)
   │
   ▼
query.py   →  ta question → récupère les chunks pertinents → Claude répond
```

## Installation

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY="ta_clé_api"
```

## Utilisation

1. **Ajoute tes documents** dans le dossier `data/` :
   - brochures / specs des logiciels (PDF)
   - notes personnelles (.txt ou .md)
   - transcriptions de démos, comparatifs déjà collectés, etc.

2. **Indexe-les** :
   ```bash
   python ingest.py
   ```
   Relance cette commande à chaque fois que tu ajoutes/modifies des documents.

3. **Pose tes questions** :
   ```bash
   python query.py "Quels outils d'estimation propose Procore par rapport à Buildertrend ?"
   python query.py "Comment fonctionne le takeoff/métré dans STACK ?"
   ```

## Personnalisation

Tout se règle dans `config.py` :
- `CHUNK_SIZE` / `CHUNK_OVERLAP` : taille des découpages (à augmenter pour des
  docs très denses en tableaux de prix)
- `TOP_K` : nombre de passages récupérés par question (monte-le si tes
  réponses manquent de contexte)
- `CLAUDE_MODEL` : modèle utilisé pour la génération
- `SYSTEM_PROMPT` : le "rôle" donné à Claude — à ajuster si tu veux un ton
  différent (ex: orienté script vidéo plutôt que réponse factuelle)

## Étapes suivantes possibles

- Ajouter un scraper pour récupérer automatiquement les pages produits des
  éditeurs (Buildertrend, Procore...) et les injecter dans `data/`
- Ajouter une interface web simple (Streamlit/Gradio) au-dessus de `query.py`
- Enrichir les métadonnées (catégorie de logiciel, date, prix mentionné) pour
  filtrer les recherches par éditeur ou fonctionnalité
