"""
04_vector_representation.py — Stage 4 of the RAG pipeline: turns chunk text
into vector embeddings using a sentence-transformers model via
langchain_huggingface.

This stage only builds the embedding MODEL — actually embedding the chunks
and storing the vectors happens in 05_create_chroma_store.py, since that's
where the vector store (which calls the embedding model internally) is
created.

Run directly for a quick smoke test:
    python 04_vector_representation.py
"""

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    return HuggingFaceEmbeddings(model_name=model_name)


if __name__ == "__main__":
    model = get_embedding_model()
    vector = model.embed_query("present perfect tense")
    print(f"Embedding dimension: {len(vector)}")
