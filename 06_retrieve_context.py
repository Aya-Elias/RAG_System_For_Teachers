"""
06_retrieve_context.py — Stage 6 of the RAG pipeline: filters chunks by
metadata (grade/section), retrieves the top-k most relevant chunks for a
query via MMR search, and assembles them into a word-budgeted context
string ready for prompting.

Run directly for a quick smoke test:
    python 06_retrieve_context.py
"""

import re
import importlib

import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS

chunking = importlib.import_module("03_chunking")
chunk_row_to_document = chunking.chunk_row_to_document

def filter_chunks_by_metadata(df, grade=None, section=None):
    subset = df
    if grade is not None and not (isinstance(grade, float) and np.isnan(grade)):
        grade_mask = subset["grade"] == grade
        if grade_mask.any():
            subset = subset[grade_mask]
    if section:
        section_mask = subset["section"] == section
        if section_mask.any():
            subset = subset[section_mask]
    return subset.reset_index(drop=True)


def retrieve_top_k(chunks_df, vectorstore, embedding_model, query, grade=None, section=None, k=5, fetch_k=20):
    filtered_df = filter_chunks_by_metadata(chunks_df, grade=grade, section=section)
    if filtered_df.empty:
        filtered_df = chunks_df
        local_store = vectorstore
    else:
        local_docs = [chunk_row_to_document(row) for _, row in filtered_df.iterrows()]
        local_store = FAISS.from_documents(local_docs, embedding_model)

    k_eff = min(k, local_store.index.ntotal)
    fetch_k_eff = min(max(fetch_k, k_eff), local_store.index.ntotal)

    retriever = local_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k_eff, "fetch_k": fetch_k_eff},
    )
    retrieved_docs = retriever.invoke(query)

    scored = local_store.similarity_search_with_relevance_scores(query, k=k_eff)
    score_by_chunk_id = {doc.metadata["chunk_id"]: score for doc, score in scored}

    rows = []
    for doc in retrieved_docs:
        rows.append({
            "chunk_id": doc.metadata["chunk_id"],
            "grade": doc.metadata["grade"],
            "section": doc.metadata["section"],
            "grammar_topic": doc.metadata["grammar_topic"],
            "literature": doc.metadata["literature"],
            "title": doc.metadata["title"],
            "chunk_text": doc.page_content,
            "score": score_by_chunk_id.get(doc.metadata["chunk_id"], 0.0),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Section 9 — Context builder
# --------------------------------------------------------------------------
def build_context(retrieved_df, word_budget=220, max_chunks=4):
    if retrieved_df.empty:
        return "", pd.DataFrame()

    ranked = retrieved_df.sort_values(by="score", ascending=False).reset_index(drop=True)

    selected_rows, seen_texts, used_words = [], set(), 0
    for _, row in ranked.iterrows():
        normalized_text = re.sub(r"\s+", " ", row["chunk_text"]).strip().lower()
        if normalized_text in seen_texts:
            continue
        chunk_words = len(row["chunk_text"].split())
        if selected_rows and used_words + chunk_words > word_budget:
            continue
        selected_rows.append(row.to_dict())
        seen_texts.add(normalized_text)
        used_words += chunk_words
        if len(selected_rows) >= max_chunks:
            break

    selected_df = pd.DataFrame(selected_rows)
    context_blocks = []
    for idx, row in enumerate(selected_rows, start=1):
        header = f"[Source {idx}] {row['title']} (Grade={row['grade']}, Section={row['section']})"
        context_blocks.append(f"{header}\n{row['chunk_text']}")
    return "\n\n".join(context_blocks), selected_df


