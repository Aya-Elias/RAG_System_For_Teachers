"""
05_create_chroma_store.py — Stage 5 of the RAG pipeline: builds the vector
store that holds the embedded chunks for similarity search.

NOTE ON THE FILENAME: the app's actual vector store is FAISS
(langchain_community.vectorstores.FAISS), not Chroma — that's a deliberate
choice made earlier in this project (FAISS needs no separate DB server/
persistence directory, which keeps deployment on Streamlit Community Cloud
simple). Per your instruction not to change anything in the app itself,
the store implementation was left as FAISS; only the filename follows the
assignment's required naming. Flag this to your professor if an actual
ChromaDB store is a hard requirement rather than an example name.

Run directly for a quick smoke test:
    python 05_create_chroma_store.py
"""

from langchain_community.vectorstores import FAISS


def create_vector_store(chunk_documents, embedding_model):
    """Embeds each chunk Document and builds the FAISS similarity index."""
    return FAISS.from_documents(chunk_documents, embedding_model)


if __name__ == "__main__":
    from langchain_core.documents import Document

    import importlib
    vector_representation = importlib.import_module("04_vector_representation")

    demo_docs = [Document(page_content="Present perfect is used for past actions with present relevance.")]
    store = create_vector_store(demo_docs, vector_representation.get_embedding_model())
    print("Vector store built with", store.index.ntotal, "vector(s).")
