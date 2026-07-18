"""
Streamlit app — Primary English Curriculum Map RAG Assistant

Rebuilds the explicit RAG pipeline from the "v3" notebook (the version that
correctly wraps prompts in the tokenizer/chat-completion format) as a
Streamlit UI, with ONE deliberate change for deployability:

    Generation calls the HuggingFace Inference API remotely (huggingface_hub
    InferenceClient) instead of downloading and running model weights locally.
    This keeps RAM/CPU usage low enough to run on Streamlit Community Cloud's
    free tier, where loading a 3B+ parameter model locally is not feasible.

Everything upstream of generation (parsing, cleaning, chunking, metadata
extraction, embeddings, FAISS retrieval, context building) is identical in
behavior to the notebook.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import docx
from docx.oxml.ns import qn
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from huggingface_hub import InferenceClient

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DOCX_PATH = "curriculum.docx"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Tried in order via the Inference API. Gated Llama models still need the
# same HF license acceptance as before -- Qwen is included as a fallback
# that needs no special access approval.
LLM_CANDIDATES = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
]

SYSTEM_PROMPT = (
    "You are an English Curriculum Assistant. Follow the instructions in the "
    "user message exactly and stay grounded in the provided context."
)

PROMPT_TEMPLATE = """You are an English Curriculum Assistant.

Answer ONLY using the context provided below. Never invent information that is not explicitly present in the context.
If the answer cannot be found in the context, reply exactly: "I couldn't find that information in the curriculum."
Mention the Grade whenever it is available in the context.
Keep your answer concise and educational, suitable for a teacher preparing a lesson.

Context:
{context}

Question:
{question}

Answer:"""


# --------------------------------------------------------------------------
# Section 2 — Document loading (unchanged from the notebook)
# --------------------------------------------------------------------------
def parse_curriculum_docx(path):
    d = docx.Document(path)
    body = d.element.body
    para_map = {p._p: p for p in d.paragraphs}
    table_map = {t._tbl: t for t in d.tables}

    seq = []
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            p = para_map.get(child)
            if p is not None:
                seq.append(("p", p.style.name if p.style else "normal", p.text))
        elif child.tag == qn("w:tbl"):
            t = table_map.get(child)
            if t is not None:
                rows = [[c.text.strip() for c in row.cells] for row in t.rows]
                seq.append(("tbl", None, rows))

    documents = []
    doc_id = 0
    i, n = 0, len(seq)
    heading_styles = ("Heading 1", "Heading 2", "Heading 3")
    lit_pattern = re.compile(r'^"([^"]+)"\s+by\s+([^:]+):\s*(.+)$')

    while i < n:
        kind, style, content = seq[i]

        if kind == "p" and style == "Heading 2" and "Executive Summary" in content:
            i += 1
            parts = []
            while i < n and seq[i][0] == "p" and seq[i][1] not in heading_styles:
                if seq[i][2].strip():
                    parts.append(seq[i][2].strip())
                i += 1
            documents.append({
                "document_id": doc_id, "grade": None, "doc_type": "overview",
                "section": "Executive Summary", "grammar_topic": None, "literature": None,
                "title": "Curriculum Executive Summary",
                "text": " ".join(parts),
            })
            doc_id += 1
            continue

        if kind == "p" and style == "Heading 3":
            m = re.match(r"Grade (\d+):\s*(.*)", content)
            grade_num = int(m.group(1)) if m else None
            fallback_title = content
            i += 1

            overview_parts = []
            while i < n and seq[i][0] == "p" and seq[i][1] not in heading_styles:
                if seq[i][2].strip():
                    overview_parts.append(seq[i][2].strip())
                i += 1
            documents.append({
                "document_id": doc_id, "grade": grade_num, "doc_type": "grade_overview",
                "section": "Grade Overview", "grammar_topic": None, "literature": None,
                "title": f"Grade {grade_num} Overview",
                "text": " ".join(overview_parts) if overview_parts else fallback_title,
            })
            doc_id += 1

            if i < n and seq[i][0] == "tbl":
                rows = seq[i][2]
                for row in rows[1:]:
                    if len(row) < 4 or not row[0].strip():
                        continue
                    concept, scope, vocab, activity = row[0], row[1], row[2], row[3]
                    documents.append({
                        "document_id": doc_id, "grade": grade_num, "doc_type": "grammar_concept",
                        "section": "Grammar Concept", "grammar_topic": concept.strip(), "literature": None,
                        "title": f"Grade {grade_num} - {concept.strip()}",
                        "text": (
                            f"Grammar concept: {concept.strip()}. "
                            f"Target scope & examples: {scope.strip()} "
                            f"Key vocabulary & signals: {vocab.strip()} "
                            f"Suggested activity: {activity.strip()}"
                        ),
                    })
                    doc_id += 1
                i += 1

            if i < n and seq[i][0] == "p" and "Assigned Stories" in seq[i][2]:
                i += 1

            while i < n and seq[i][0] == "p" and seq[i][1] not in heading_styles:
                text = seq[i][2].strip()
                if not text:
                    i += 1
                    continue
                lm = lit_pattern.match(text)
                if not lm:
                    break
                book_title, author, usage = lm.group(1), lm.group(2).strip(), lm.group(3).strip()
                documents.append({
                    "document_id": doc_id, "grade": grade_num, "doc_type": "literature",
                    "section": "Assigned Literature", "grammar_topic": None, "literature": book_title,
                    "title": f"Grade {grade_num} - {book_title}",
                    "text": f'Assigned story: "{book_title}" by {author}. {usage}',
                })
                doc_id += 1
                i += 1
            continue

        if kind == "p" and style == "Heading 2" and "Quick Reference Matrix" in content:
            i += 1
            intro_parts = []
            while i < n and seq[i][0] == "p" and seq[i][1] not in heading_styles:
                if seq[i][2].strip():
                    intro_parts.append(seq[i][2].strip())
                i += 1
            if i < n and seq[i][0] == "tbl":
                rows = seq[i][2]
                lines = []
                for row in rows[1:]:
                    if len(row) < 3 or not row[0].strip():
                        continue
                    lines.append(f"{row[0].strip()}: {row[1].strip()} — literature: {row[2].strip()}")
                documents.append({
                    "document_id": doc_id, "grade": None, "doc_type": "reference_matrix",
                    "section": "Reference Matrix", "grammar_topic": None, "literature": None,
                    "title": "Quick Reference Matrix",
                    "text": " ".join(intro_parts) + " " + " | ".join(lines),
                })
                doc_id += 1
                i += 1
            continue

        i += 1

    df = pd.DataFrame(documents)
    if df.empty:
        raise ValueError(
            "No documents were extracted. The .docx structure may not match the "
            "expected headings (Heading 2 'Executive Summary' / Heading 3 'Grade N' / "
            "Heading 2 'Quick Reference Matrix')."
        )
    return df


# --------------------------------------------------------------------------
# Section 3 — Cleaning
# --------------------------------------------------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[-_=]{3,}", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


# --------------------------------------------------------------------------
# Section 4 — Adaptive chunking
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Section 8 — Metadata filtering + MMR retrieval
# --------------------------------------------------------------------------
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


def build_grounded_prompt(query, context_text):
    if not context_text:
        context_text = "(No relevant context was retrieved.)"
    return PROMPT_TEMPLATE.format(context=context_text, question=query)


def extract_final_answer(raw_text):
    if not raw_text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned).strip()
    return cleaned


# --------------------------------------------------------------------------
# Generation via HF Inference API (remote — no local weights)
# --------------------------------------------------------------------------
def generate_answer(prompt, hf_token):
    """Try each candidate model via the Inference API chat-completion
    endpoint until one succeeds. Returns (answer_text, model_used_or_None)."""
    if not hf_token:
        return (
            "[No LLM response — no HuggingFace token was provided. "
            "Add HF_TOKEN in the sidebar or in Streamlit secrets.]",
            None,
        )

    last_error = None
    for candidate in LLM_CANDIDATES:
        try:
            client = InferenceClient(model=candidate, token=hf_token)
            completion = client.chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=250,
                temperature=0.0,
            )
            answer = completion.choices[0].message.content
            return extract_final_answer(answer), candidate
        except Exception as exc:
            last_error = exc
            continue

    return (
        f"[No LLM response — all candidate models failed. "
        f"Last error: {last_error}]",
        None,
    )


# --------------------------------------------------------------------------
# Cached pipeline setup (runs once per session, not on every question)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Building the curriculum index…")
def build_index(docx_path):
    documents_df = parse_curriculum_docx(docx_path)
    documents_df["text"] = documents_df["text"].map(clean_text)
    chunks_df = build_adaptive_chunks(documents_df)

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    all_chunk_documents = [chunk_row_to_document(row) for _, row in chunks_df.iterrows()]
    vectorstore = FAISS.from_documents(all_chunk_documents, embedding_model)

    return chunks_df, vectorstore, embedding_model


# --------------------------------------------------------------------------
# Guided question builder — lets the user pick each part of the question
# from dropdowns (populated from the actual curriculum content) instead of
# typing free text, then composes a natural-language question from it.
# --------------------------------------------------------------------------
QUESTION_TYPES = {
    "قاعدة نحوية معيّنة (Grammar concept)": "grammar",
    "قصة أو كتاب معيّن (Literature)": "literature",
    "نظرة عامة على صف دراسي (Grade overview)": "overview",
    "ملخص المنهج العام (Executive summary)": "summary",
}


def build_guided_question(chunks_df):
    """Renders cascading selectboxes and returns (question_text, grade_value)."""
    type_label = st.selectbox("١) عايزة تعرفي إيه؟", list(QUESTION_TYPES.keys()))
    qtype = QUESTION_TYPES[type_label]

    if qtype == "grammar":
        grades = sorted(chunks_df.loc[chunks_df["doc_type"] == "grammar_concept", "grade"].dropna().unique())
        grade = st.selectbox("٢) الصف الدراسي", [int(g) for g in grades])
        topics = sorted(
            chunks_df.loc[
                (chunks_df["doc_type"] == "grammar_concept") & (chunks_df["grade"] == grade),
                "grammar_topic",
            ].dropna().unique()
        )
        topic = st.selectbox("٣) القاعدة النحوية", topics)
        return f"How do I teach {topic} to grade {grade} students?", grade

    if qtype == "literature":
        grades = sorted(chunks_df.loc[chunks_df["doc_type"] == "literature", "grade"].dropna().unique())
        grade_choice = st.selectbox("٢) الصف الدراسي (اختياري)", ["أي صف"] + [int(g) for g in grades])
        grade = None if grade_choice == "أي صف" else grade_choice
        lit_pool = chunks_df[chunks_df["doc_type"] == "literature"]
        if grade is not None:
            lit_pool = lit_pool[lit_pool["grade"] == grade]
        books = sorted(lit_pool["literature"].dropna().unique())
        book = st.selectbox("٣) القصة/الكتاب", books)
        if grade is not None:
            return f'How is "{book}" used to teach English in grade {grade}?', grade
        return f'Which grade uses "{book}" and how is it taught?', None

    if qtype == "overview":
        grades = sorted(chunks_df.loc[chunks_df["doc_type"] == "grade_overview", "grade"].dropna().unique())
        grade = st.selectbox("٢) الصف الدراسي", [int(g) for g in grades])
        return f"What is the overall focus of grade {grade} curriculum?", grade

    # summary
    return "What is the executive summary of the curriculum?", None


# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
st.set_page_config(page_title="Curriculum RAG Assistant", page_icon="📚", layout="centered")
st.title("📚 English Curriculum Assistant")
st.caption("Ask a question about the Primary English Curriculum Map (Grades 1–6). Answers are grounded only in the curriculum document.")

with st.sidebar:
    st.header("Settings")
    hf_token = st.text_input(
        "HuggingFace Token",
        value=os.environ.get("HF_TOKEN", st.secrets.get("HF_TOKEN", "") if hasattr(st, "secrets") else ""),
        type="password",
        help="Needed to call the HuggingFace Inference API for generation. Get one at huggingface.co/settings/tokens.",
    )
    grade_filter = st.selectbox("Grade filter (optional)", ["Any"] + [str(g) for g in range(1, 7)])
    grade_value = None if grade_filter == "Any" else int(grade_filter)
    k = st.slider("Chunks to retrieve (k)", min_value=2, max_value=8, value=5)
    st.divider()
    st.caption("Generation runs remotely via the HF Inference API — no model weights are downloaded on this server.")

if not Path(DOCX_PATH).exists():
    st.error(f"'{DOCX_PATH}' not found next to app.py. Add the curriculum file to the repo.")
    st.stop()

chunks_df, vectorstore, embedding_model = build_index(DOCX_PATH)

mode = st.radio("طريقة السؤال", ["اختاري من قوائم (موجّه)", "اكتبي سؤال حر"], horizontal=True)

guided_grade = None
if mode == "اختاري من قوائم (موجّه)":
    question, guided_grade = build_guided_question(chunks_df)
    st.text_input("السؤال اللي هيتبعت", value=question, disabled=True)
else:
    question = st.text_input("Your question", placeholder="e.g. How do I teach present perfect to grade 5 students?")

ask = st.button("Ask", type="primary")

# In guided mode, the grade picked while building the question takes
# priority over the sidebar filter (it's more specific to this question).
effective_grade = guided_grade if mode == "اختاري من قوائم (موجّه)" and guided_grade is not None else grade_value

if ask and question.strip():
    with st.spinner("Retrieving context and generating an answer…"):
        retrieved = retrieve_top_k(
            chunks_df, vectorstore, embedding_model, question,
            grade=effective_grade, k=k, fetch_k=max(20, k * 4),
        )
        context_text, selected_df = build_context(retrieved)
        prompt = build_grounded_prompt(question, context_text)
        answer, model_used = generate_answer(prompt, hf_token)

    st.subheader("Answer")
    st.write(answer)

    if model_used:
        st.caption(f"Generated with: {model_used}")

    if not selected_df.empty:
        st.subheader("Sources")
        for _, row in selected_df.iterrows():
            with st.expander(f"{row['title']}  ·  score={row['score']:.3f}"):
                st.write(row["chunk_text"])
    else:
        st.info("No matching chunks were retrieved for this question.")
elif ask:
    st.warning("Type a question first.")
