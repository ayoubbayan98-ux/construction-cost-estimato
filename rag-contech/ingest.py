"""
Ingestion : lit tous les documents du dossier data/ (PDF, TXT, MD),
les découpe en chunks, calcule leurs embeddings, et les stocke dans ChromaDB.

Usage :
    python ingest.py
"""

import hashlib
import re
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import config


TOOL_ALIASES = {
    "buildertrend": "Buildertrend",
    "procore": "Procore",
    "planswift": "PlanSwift",
    "stack": "STACK",
    "proest": "ProEst",
    "rsmeans": "RSMeans",
}

CATEGORY_KEYWORDS = {
    "estimation": ["estimat", "takeoff", "métré", "pricing", "cost"],
    "planning": ["planning", "schedule", "chantier"],
    "budget": ["budget", "suivi", "finance"],
}


def extract_front_matter_metadata(text: str) -> dict:
    """Retourne les métadonnées YAML-like entre les marqueurs --- ."""
    if not text.startswith("---"):
        return {}
    try:
        _, front_matter, _ = text.split("---\n", 2)
    except ValueError:
        return {}

    metadata = {}
    for line in front_matter.splitlines():
        if not line.strip() or ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip().lower()] = value.strip()
    return metadata


def infer_document_metadata(path: Path | str, front_matter: dict | None = None) -> dict:
    """Déduit des métadonnées à partir du nom du fichier et du front matter."""
    path_obj = Path(path)
    source = path_obj.name if hasattr(path_obj, "name") else str(path)
    file_name = source.lower()
    metadata = {"source": source, "tool": None, "category": "general", "cost_band": "unknown"}

    if front_matter:
        metadata.update({
            "tool": front_matter.get("tool") or metadata["tool"],
            "category": front_matter.get("category") or metadata["category"],
            "cost_band": front_matter.get("cost_band") or metadata["cost_band"],
        })

    for alias, canonical in TOOL_ALIASES.items():
        if alias in file_name:
            metadata["tool"] = canonical
            break

    if metadata["category"] == "general":
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword in file_name for keyword in keywords):
                metadata["category"] = category
                break

    if metadata["cost_band"] == "unknown":
        if any(token in file_name for token in ("low", "budget")):
            metadata["cost_band"] = "low"
        elif any(token in file_name for token in ("medium", "standard")):
            metadata["cost_band"] = "medium"
        elif any(token in file_name for token in ("high", "premium")):
            metadata["cost_band"] = "high"

    return metadata


def read_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(data_dir: Path) -> list[dict]:
    """Charge les fichiers supportés avec leurs métadonnées documentaires."""
    docs = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_dir():
            continue
        if path.suffix.lower() == ".pdf":
            text = read_pdf(path)
        elif path.suffix.lower() in (".txt", ".md"):
            text = read_text_file(path)
        else:
            continue
        if not text.strip():
            continue
        front_matter = extract_front_matter_metadata(text)
        metadata = infer_document_metadata(path, front_matter)
        docs.append({"source": path.name, "text": text, "metadata": metadata})
    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Découpage simple en chunks avec chevauchement."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


def make_chunk_id(source: str, index: int, text: str) -> str:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{source}-{index}-{h}"


def main():
    config.DATA_DIR.mkdir(exist_ok=True)
    config.DB_DIR.mkdir(exist_ok=True)

    docs = load_documents(config.DATA_DIR)
    if not docs:
        print(f"Aucun document trouvé dans {config.DATA_DIR}.")
        print("Mets des fichiers .pdf, .txt ou .md dedans puis relance ce script.")
        return

    print(f"{len(docs)} document(s) trouvé(s). Chunking...")
    all_chunks, all_ids, all_metadatas = [], [], []
    for doc in docs:
        chunks = chunk_text(doc["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(make_chunk_id(doc["source"], i, chunk))
            metadata = {
                "source": doc["source"],
                "tool": doc["metadata"].get("tool"),
                "category": doc["metadata"].get("category"),
                "cost_band": doc["metadata"].get("cost_band"),
                "chunk_index": i,
            }
            all_metadatas.append(metadata)

    print(f"{len(all_chunks)} chunks au total. Calcul des embeddings...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    collection = client.get_or_create_collection(config.COLLECTION_NAME)

    print("Sauvegarde dans ChromaDB...")
    batch_size = 100
    for i in tqdm(range(0, len(all_chunks), batch_size)):
        collection.upsert(
            ids=all_ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            documents=all_chunks[i:i + batch_size],
            metadatas=all_metadatas[i:i + batch_size],
        )

    print(f"Terminé. {len(all_chunks)} chunks indexés dans '{config.COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
