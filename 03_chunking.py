"""
03_chunking.py — Stage 3 of the RAG pipeline: splits each cleaned
document into retrieval-sized chunks. Grammar-concept and literature
rows are kept whole (they are already short, self-contained units);
everything else is split by sentence boundaries up to a max word budget.

Run directly for a quick smoke test with a tiny fake documents frame:
    python 03_chunking.py
"""

import re

import pandas as pd
from langchain_core.documents import Document

CONCEPT_TYPES = {"grade_overview", "grammar_concept", "literature"}

MAX_WORDS_PER_CHUNK = 160


def split_by_structure(text, max_words=MAX_WORDS_PER_CHUNK):
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    parts, current, count = [], [], 0
    for s in sentences:
        w = len(s.split())
        if current and count + w > max_words:
            parts.append(" ".join(current))
            current, count = [], 0
        current.append(s)
        count += w
    if current:
        parts.append(" ".join(current))
    return parts if parts else [text]


def build_adaptive_chunks(documents_frame):
    rows = []
    for _, doc in documents_frame.iterrows():
        if doc["doc_type"] in CONCEPT_TYPES:
            pieces = [doc["text"]]
        else:
            pieces = split_by_structure(doc["text"])
        for idx, piece in enumerate(pieces):
            rows.append({
                "chunk_id": f"doc{doc['document_id']}_chunk{idx}",
                "document_id": doc["document_id"],
                "grade": doc["grade"],
                "section": doc["section"],
                "doc_type": doc["doc_type"],
                "grammar_topic": doc["grammar_topic"],
                "literature": doc["literature"],
                "title": doc["title"],
                "chunk_text": piece,
                "word_count": len(piece.split()),
            })
    return pd.DataFrame(rows)


def chunk_row_to_document(row):
    return Document(
        page_content=row["chunk_text"],
        metadata={
            "chunk_id": row["chunk_id"],
            "grade": row["grade"],
            "section": row["section"],
            "grammar_topic": row["grammar_topic"],
            "literature": row["literature"],
            "title": row["title"],
        },
    )


if __name__ == "__main__":
    demo_df = pd.DataFrame([{
        "document_id": 1, "grade": 3, "section": "grammar", "doc_type": "grammar_concept",
        "grammar_topic": "Present Perfect", "literature": None, "title": "Grade 3 - Present Perfect",
        "text": "Present perfect is used for past actions with present relevance.",
    }])
    chunks = build_adaptive_chunks(demo_df)
    print(chunks)
