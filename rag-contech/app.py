import os
from typing import Optional

import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer

import config
import ingest
from query import answer_from_context, ask_claude, build_context, build_where_clause, retrieve


st.set_page_config(page_title="RAG Contech", page_icon="🏗️", layout="wide")


@st.cache_resource
def get_model():
    return SentenceTransformer(config.EMBEDDING_MODEL)


@st.cache_resource
def get_collection():
    config.DATA_DIR.mkdir(exist_ok=True)
    config.DB_DIR.mkdir(exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    try:
        return client.get_collection(config.COLLECTION_NAME)
    except Exception:
        ingest.main()
        return client.get_collection(config.COLLECTION_NAME)


def save_uploaded_files(uploaded_files):
    if not uploaded_files:
        return []

    saved = []
    for uploaded_file in uploaded_files:
        target_path = config.DATA_DIR / uploaded_file.name
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        saved.append(str(target_path))
    return saved


def search(question: str, tool: Optional[str], category: Optional[str], cost: Optional[str], source: Optional[str]):
    model = get_model()
    collection = get_collection()
    filters = build_where_clause(source=source, tool=tool, category=category, cost=cost)
    retrieved = retrieve(question, model, collection, config.TOP_K, filters)
    context = build_context(retrieved)
    return retrieved, context


st.markdown(
    """
    <style>
    .main {
        background: #f5f7fb;
    }
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    div[data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏗️ RAG Contech")
st.caption("Assistant de recherche documentaire pour l’estimation de coûts de construction")

with st.sidebar:
    st.header("📁 Documents")
    uploaded_files = st.file_uploader(
        "Télécharger des PDF / TXT / MD",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Indexer les fichiers uploadés", use_container_width=True):
        if uploaded_files:
            with st.spinner("Sauvegarde et indexation des fichiers..."):
                saved = save_uploaded_files(uploaded_files)
                ingest.main()
                st.success(f"{len(saved)} fichier(s) ajouté(s) et indexé(s).")
        else:
            st.warning("Aucun fichier sélectionné.")

    st.header("🔎 Filtres")
    source = st.text_input("Nom de fichier / source", placeholder="stack_estimation.md")
    tool = st.selectbox("Outil", ["", "Buildertrend", "Procore", "STACK", "PlanSwift", "ProEst", "RSMeans"], index=0)
    category = st.selectbox("Catégorie", ["", "estimation", "planning", "budget", "general"], index=0)
    cost = st.selectbox("Niveau de coût", ["", "low", "medium", "high", "unknown"], index=0)

    if st.button("Reindexer la base", use_container_width=True):
        with st.spinner("Réindexation complète du dossier data/..."):
            ingest.main()
            st.success("Base réindexée.")

col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area(
        "Question",
        value="Quel outil est spécialisé dans le takeoff numérique ?",
        height=140,
    )

with col2:
    st.markdown("### Démo")
    st.write("- Buildertrend")
    st.write("- Procore")
    st.write("- STACK")
    st.write("- PlanSwift")
    st.write("- RSMeans")

if st.button("Rechercher", use_container_width=True) and question.strip():
    with st.spinner("Recherche dans l’index et génération de la réponse..."):
        retrieved, context = search(question, tool or None, category or None, cost or None, source or None)

        if not retrieved:
            st.warning("Aucun résultat ne correspond aux filtres demandés.")
            st.stop()

        st.subheader("📚 Sources retrouvées")
        sources = sorted({meta.get("source", "inconnu") for _, meta in retrieved})
        st.write(", ".join(sources))

        if config.ANTHROPIC_API_KEY:
            try:
                from anthropic import Anthropic

                client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
                answer = ask_claude(client, question, context)
            except Exception as exc:
                st.warning(f"Échec de l’API Anthropic ({exc}). Utilisation du mode hors ligne.")
                answer = answer_from_context(question, context)
        else:
            st.info("Aucune clé Anthropic détectée. Mode hors ligne activé.")
            answer = answer_from_context(question, context)

        st.subheader("💬 Réponse")
        st.write(answer)

        with st.expander("Contexte récupéré"):
            st.text(context)
else:
    st.info("Saisis une question et clique sur Rechercher.")
