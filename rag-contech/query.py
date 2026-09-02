"""
Interroge le RAG : récupère les chunks pertinents dans ChromaDB
puis demande à Claude de répondre en se basant dessus.

Usage :
    python query.py "Quelle est la différence entre Buildertrend et Procore pour l'estimation ?"
    python query.py "Quel outil est spécialisé dans le takeoff ?" --tool STACK --category estimation
"""

import argparse
import sys

import chromadb
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

import config


def build_where_clause(source=None, tool=None, category=None, cost=None):
    filters = []
    if source:
        filters.append({"source": {"$eq": source}})
    if tool:
        filters.append({"tool": {"$eq": tool}})
    if category:
        filters.append({"category": {"$eq": category}})
    if cost:
        filters.append({"cost_band": {"$eq": cost}})
    return {"$and": filters} if filters else None


def retrieve(question: str, model: SentenceTransformer, collection, top_k: int, where_clause=None):
    query_embedding = model.encode([question]).tolist()
    params = {"query_embeddings": query_embedding, "n_results": top_k}
    if where_clause:
        params["where"] = where_clause
    results = collection.query(**params)
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    return list(zip(chunks, metadatas))


def build_context(retrieved: list[tuple[str, dict]]) -> str:
    parts = []
    for text, meta in retrieved:
        source = meta.get("source", "inconnu")
        tool = meta.get("tool")
        category = meta.get("category")
        cost_band = meta.get("cost_band")
        label = f"[Source: {source}"
        if tool:
            label += f" | Outil: {tool}"
        if category:
            label += f" | Catégorie: {category}"
        if cost_band:
            label += f" | Coût: {cost_band}"
        label += "]"
        parts.append(f"{label}\n{text}")
    return "\n\n---\n\n".join(parts)


def answer_from_context(question: str, context: str) -> str:
    """Fallback local simple : synthétise une réponse à partir du contexte récupéré."""
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    snippets = []
    for line in lines[:12]:
        if line.startswith("[Source:"):
            continue
        snippets.append(line)

    summary = "\n".join(snippets[:8])
    if not summary:
        return "Je n’ai pas trouvé de contexte exploitable dans l’index pour répondre à cette question."

    return (
        "Voici ce que j’ai trouvé dans les documents indexés :\n\n"
        f"{summary}\n\n"
        "Cette réponse est produite en mode hors ligne, sans clé API Anthropic. "
        "Pour un résumé plus avancé, ajoute ANTHROPIC_API_KEY et relance la commande."
    )


def ask_claude(client: Anthropic, question: str, context: str) -> str:
    message = client.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=1500,
        system=config.SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Contexte extrait des documents :\n\n{context}\n\n---\n\nQuestion : {question}",
            }
        ],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Query the contech RAG with optional metadata filters.")
    parser.add_argument("question", nargs="*", help="Question to ask the RAG")
    parser.add_argument("--source", help="Filtrer sur un nom de document exact")
    parser.add_argument("--tool", help="Filtrer sur un outil (ex: Procore, STACK, Buildertrend)")
    parser.add_argument("--category", help="Filtrer sur une catégorie (ex: estimation, planning, budget)")
    parser.add_argument("--cost", help="Filtrer sur un niveau de coût (ex: low, medium, high)")
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    question = " ".join(args.question)
    if not question:
        print('Usage : python query.py "ta question ici" [--tool Procore] [--category estimation] [--cost medium]')
        return

    print("Chargement du modèle d'embeddings...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    client_db = chromadb.PersistentClient(path=str(config.DB_DIR))
    try:
        collection = client_db.get_collection(config.COLLECTION_NAME)
    except Exception:
        print("Aucune collection trouvée. Lance d'abord : python ingest.py")
        return

    filters = build_where_clause(
        source=args.source,
        tool=args.tool,
        category=args.category,
        cost=args.cost,
    )
    if filters:
        print(f"Filtres actifs : {filters}")

    print(f"Recherche des {config.TOP_K} passages les plus pertinents...")
    retrieved = retrieve(question, model, collection, config.TOP_K, filters)

    if not retrieved:
        print("Aucun résultat ne correspond aux filtres demandés.")
        return

    print("Sources utilisées :", ", ".join(sorted({m["source"] for _, m in retrieved})))
    context = build_context(retrieved)

    print("\nGénération de la réponse...\n")
    if not config.ANTHROPIC_API_KEY:
        print("⚠️  Aucune clé Anthropic détectée. Utilisation du mode hors ligne.\n")
        answer = answer_from_context(question, context)
    else:
        try:
            client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
            answer = ask_claude(client, question, context)
        except Exception as exc:
            print(f"⚠️  Échec de l’API Anthropic ({exc}). Utilisation du mode hors ligne.\n")
            answer = answer_from_context(question, context)

    print("=" * 60)
    print(answer)
    print("=" * 60)


if __name__ == "__main__":
    main()
