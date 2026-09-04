"""
Interface Streamlit pour interroger le RAG contech (ChromaDB + Claude).

Usage :
    streamlit run streamlit_app.py
"""

import streamlit as st
import chromadb
from anthropic import Anthropic
from sentence_transformers import SentenceTransformer

import config
from query import (
    build_where_clause,
    retrieve,
    build_context,
    ask_claude,
    answer_from_context,
)

st.set_page_config(page_title="ConTech RAG", page_icon="🏗️", layout="centered")


@st.cache_resource(show_spinner="Chargement du modèle d'embeddings...")
def load_model():
    return SentenceTransformer(config.EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Connexion à la base ChromaDB...")
def load_collection():
    client_db = chromadb.PersistentClient(path=str(config.DB_DIR))
    try:
        return client_db.get_collection(config.COLLECTION_NAME)
    except Exception:
        return None


model = load_model()
collection = load_collection()

st.title("🏗️ ConTech RAG — Assistant Estimation")
st.caption("Pose une question sur les outils de construction tech (Procore, STACK, Buildertrend...)")

if collection is None:
    st.error(
        "Aucune collection trouvée dans ChromaDB. "
        "Lance d'abord `python ingest.py` dans le terminal pour indexer les documents."
    )
    st.stop()

with st.sidebar:
    st.header("Filtres (optionnel)")
    tool = st.text_input("Outil (ex: Procore, STACK, Buildertrend)")
    category = st.text_input("Catégorie (ex: estimation, planning, budget)")
    cost = st.text_input("Niveau de coût (ex: low, medium, high)")
    source = st.text_input("Nom de document exact")
    top_k = st.slider("Nombre de passages à récupérer", min_value=1, max_value=10, value=config.TOP_K)

question = st.text_area("Ta question", placeholder="Quelle est la différence entre Buildertrend et Procore pour l'estimation ?")

if st.button("Envoyer", type="primary") and question.strip():
    filters = build_where_clause(
        source=source or None,
        tool=tool or None,
        category=category or None,
        cost=cost or None,
    )

    with st.spinner("Recherche des passages pertinents..."):
        retrieved = retrieve(question, model, collection, top_k, filters)

    if not retrieved:
        st.warning("Aucun résultat ne correspond à ta question ou tes filtres.")
    else:
        sources_used = sorted({m["source"] for _, m in retrieved})
        st.info("Sources utilisées : " + ", ".join(sources_used))

        context = build_context(retrieved)

        with st.spinner("Génération de la réponse..."):
            if not config.ANTHROPIC_API_KEY:
                answer = answer_from_context(question, context)
                st.warning("⚠️ Aucune clé ANTHROPIC_API_KEY détectée — réponse en mode hors ligne.")
            else:
                try:
                    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
                    answer = ask_claude(client, question, context)
                except Exception as exc:
                    st.warning(f"⚠️ Échec de l'API Anthropic ({exc}) — bascule en mode hors ligne.")
                    answer = answer_from_context(question, context)

        st.subheader("Réponse")
        st.markdown(answer)

        with st.expander("Voir le contexte utilisé"):
            st.text(context)
