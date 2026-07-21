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
# Section 2 — Document loading
#
# Handles TWO possible curriculum.docx layouts, since the source document
# has changed format over time:
#   (a) table-based: grammar concepts / reference matrix rows live inside
#       Word tables directly under the "Grade N" / "Quick Reference Matrix"
#       headings.
#   (b) paragraph-based: no tables at all. Grammar concepts sit under a
#       "Curricular Breakdown:" (Heading 4) as repeating groups of plain
#       paragraphs ("Grammar Concept: ...", "Target Scope & Examples: ...",
#       "Key Vocabulary & Signals: ...", "Suggested Activity: ..."),
#       literature sits under an "Assigned Stories & Literature:"
#       (Heading 4), and the reference matrix is repeating paragraph
#       triples ("Grade N Focus:", "Key Grammar Tense / Concept Focus: ...",
#       "Core Assigned Literary Work(s): ...") instead of a table.
# --------------------------------------------------------------------------
ALL_HEADING_STYLES = ("Heading 1", "Heading 2", "Heading 3", "Heading 4")
TOP_HEADING_STYLES = ("Heading 1", "Heading 2", "Heading 3")

CONCEPT_RE = re.compile(r"^Grammar Concept:\s*(.+)$")
SCOPE_RE = re.compile(r"^Target Scope\s*&\s*Examples:\s*(.+)$")
VOCAB_RE = re.compile(r"^Key Vocabulary\s*&\s*Signals:\s*(.+)$")
ACTIVITY_RE = re.compile(r"^Suggested Activity:\s*(.+)$")
LIT_PATTERN = re.compile(r'^"([^"]+)"\s+by\s+([^:]+):\s*(.+)$')
FOCUS_GRADE_RE = re.compile(r"^Grade (\d+) Focus:$")
FOCUS_CONCEPT_RE = re.compile(r"^Key Grammar Tense\s*/\s*Concept Focus:\s*(.+)$")
FOCUS_LIT_RE = re.compile(r"^Core Assigned Literary Work\(s\):\s*(.+)$")


def _consume_plain_paragraphs(seq, i, n, stop_styles=TOP_HEADING_STYLES):
    """Collects consecutive normal paragraphs, stopping at any heading in stop_styles."""
    parts = []
    while i < n and seq[i][0] == "p" and seq[i][1] not in stop_styles:
        if seq[i][2].strip():
            parts.append(seq[i][2].strip())
        i += 1
    return parts, i


def _parse_grammar_concepts_from_paragraphs(seq, i, n):
    """Parses repeating 'Grammar Concept: / Target Scope & Examples: / Key
    Vocabulary & Signals: / Suggested Activity:' paragraph groups until the
    next heading. Returns (list_of_concept_dicts, new_index)."""
    concepts = []
    current = None
    while i < n and seq[i][0] == "p" and seq[i][1] not in ALL_HEADING_STYLES:
        text = seq[i][2].strip()
        i += 1
        if not text:
            continue
        cm = CONCEPT_RE.match(text)
        if cm:
            if current:
                concepts.append(current)
            current = {"concept": cm.group(1).strip(), "scope": "", "vocab": "", "activity": ""}
            continue
        if current is None:
            continue
        sm = SCOPE_RE.match(text)
        if sm:
            current["scope"] = sm.group(1).strip()
            continue
        vm = VOCAB_RE.match(text)
        if vm:
            current["vocab"] = vm.group(1).strip()
            continue
        am = ACTIVITY_RE.match(text)
        if am:
            current["activity"] = am.group(1).strip()
    if current:
        concepts.append(current)
    return concepts, i


def _parse_literature_from_paragraphs(seq, i, n):
    """Parses repeating '"Title" by Author: usage' lines until a non-matching
    line or a heading. Returns (list_of_(title, author, usage), new_index)."""
    books = []
    while i < n and seq[i][0] == "p" and seq[i][1] not in ALL_HEADING_STYLES:
        text = seq[i][2].strip()
        if not text:
            i += 1
            continue
        lm = LIT_PATTERN.match(text)
        if not lm:
            break
        books.append((lm.group(1), lm.group(2).strip(), lm.group(3).strip()))
        i += 1
    return books, i


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

    while i < n:
        kind, style, content = seq[i]

        if kind == "p" and style == "Heading 2" and "Executive Summary" in content:
            i += 1
            parts, i = _consume_plain_paragraphs(seq, i, n)
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

            # Overview text: plain paragraphs immediately after the grade
            # heading, stopping at ANY heading level (including Heading 4
            # subsections like "Curricular Breakdown:").
            overview_parts, i = _consume_plain_paragraphs(seq, i, n, stop_styles=ALL_HEADING_STYLES)
            documents.append({
                "document_id": doc_id, "grade": grade_num, "doc_type": "grade_overview",
                "section": "Grade Overview", "grammar_topic": None, "literature": None,
                "title": f"Grade {grade_num} Overview",
                "text": " ".join(overview_parts) if overview_parts else fallback_title,
            })
            doc_id += 1

            # --- Grammar concepts: table format OR Heading-4 paragraph format ---
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
            elif i < n and seq[i][0] == "p" and seq[i][1] == "Heading 4" and "Curricular Breakdown" in seq[i][2]:
                i += 1
                concepts, i = _parse_grammar_concepts_from_paragraphs(seq, i, n)
                for c in concepts:
                    documents.append({
                        "document_id": doc_id, "grade": grade_num, "doc_type": "grammar_concept",
                        "section": "Grammar Concept", "grammar_topic": c["concept"], "literature": None,
                        "title": f"Grade {grade_num} - {c['concept']}",
                        "text": (
                            f"Grammar concept: {c['concept']}. "
                            f"Target scope & examples: {c['scope']} "
                            f"Key vocabulary & signals: {c['vocab']} "
                            f"Suggested activity: {c['activity']}"
                        ),
                    })
                    doc_id += 1

            # --- Assigned literature: legacy inline marker OR Heading-4 marker ---
            if i < n and seq[i][0] == "p" and seq[i][1] == "Heading 4" and "Assigned Stories" in seq[i][2]:
                i += 1
            elif i < n and seq[i][0] == "p" and seq[i][1] not in ALL_HEADING_STYLES and "Assigned Stories" in seq[i][2]:
                i += 1

            books, i = _parse_literature_from_paragraphs(seq, i, n)
            for book_title, author, usage in books:
                documents.append({
                    "document_id": doc_id, "grade": grade_num, "doc_type": "literature",
                    "section": "Assigned Literature", "grammar_topic": None, "literature": book_title,
                    "title": f"Grade {grade_num} - {book_title}",
                    "text": f'Assigned story: "{book_title}" by {author}. {usage}',
                })
                doc_id += 1
            continue

        if kind == "p" and style == "Heading 2" and "Quick Reference Matrix" in content:
            i += 1
            intro_parts = []
            while i < n and seq[i][0] == "p" and seq[i][1] not in ALL_HEADING_STYLES:
                text = seq[i][2].strip()
                if FOCUS_GRADE_RE.match(text):
                    break  # matrix rows start here — stop treating text as intro
                if text:
                    intro_parts.append(text)
                i += 1

            lines = []
            if i < n and seq[i][0] == "tbl":
                rows = seq[i][2]
                for row in rows[1:]:
                    if len(row) < 3 or not row[0].strip():
                        continue
                    lines.append(f"{row[0].strip()}: {row[1].strip()} — literature: {row[2].strip()}")
                i += 1
            else:
                # Paragraph-triple format: "Grade N Focus:" / "Key Grammar
                # Tense / Concept Focus: ..." / "Core Assigned Literary Work(s): ..."
                pending_grade, pending_concept = None, None
                while i < n and seq[i][0] == "p" and seq[i][1] not in TOP_HEADING_STYLES:
                    text = seq[i][2].strip()
                    i += 1
                    if not text:
                        continue
                    gm = FOCUS_GRADE_RE.match(text)
                    if gm:
                        pending_grade, pending_concept = gm.group(1), None
                        continue
                    cm = FOCUS_CONCEPT_RE.match(text)
                    if cm and pending_grade is not None:
                        pending_concept = cm.group(1).strip()
                        continue
                    lm = FOCUS_LIT_RE.match(text)
                    if lm and pending_grade is not None:
                        lines.append(f"Grade {pending_grade}: {pending_concept or ''} — literature: {lm.group(1).strip()}")
                        pending_grade, pending_concept = None, None

            documents.append({
                "document_id": doc_id, "grade": None, "doc_type": "reference_matrix",
                "section": "Reference Matrix", "grammar_topic": None, "literature": None,
                "title": "Quick Reference Matrix",
                "text": " ".join(intro_parts) + " " + " | ".join(lines),
            })
            doc_id += 1
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
    "Find which grade teaches a topic": "topic_lookup",
    "Find which grade uses a book/story": "book_lookup",
    "Find the activity for a topic in a grade": "activity",
    "List grammar topics taught in a grade": "grammar_list",
    "List stories/books taught in a grade": "book_list",
    "See everything covered in a grade": "overview",
}


def build_guided_question(chunks_df):
    """Renders cascading selectboxes and returns (question_text, grade_value).

    Framed around what a new teacher actually needs: locating which grade/
    stage a curriculum item belongs to, or finding the suggested activity
    for a topic they already know the grade of.
    """
    type_label = st.selectbox("1) What do you want to know?", list(QUESTION_TYPES.keys()))
    qtype = QUESTION_TYPES[type_label]

    if qtype == "topic_lookup":
        topics = sorted(
            chunks_df.loc[chunks_df["doc_type"] == "grammar_concept", "grammar_topic"].dropna().unique()
        )
        topic = st.selectbox("2) Grammar topic", topics)
        return f"Which grade teaches {topic}, and what stage is it introduced at?", None

    if qtype == "book_lookup":
        books = sorted(chunks_df.loc[chunks_df["doc_type"] == "literature", "literature"].dropna().unique())
        book = st.selectbox("2) Book / story", books)
        return f'Which grade uses "{book}" and at what stage is it assigned?', None

    if qtype == "activity":
        grades = sorted(chunks_df.loc[chunks_df["doc_type"] == "grammar_concept", "grade"].dropna().unique())
        grade = st.selectbox("2) Grade", [int(g) for g in grades])
        topics = sorted(
            chunks_df.loc[
                (chunks_df["doc_type"] == "grammar_concept") & (chunks_df["grade"] == grade),
                "grammar_topic",
            ].dropna().unique()
        )
        topic = st.selectbox("3) Topic", topics)
        return f"What classroom activity is suggested for teaching {topic} in grade {grade}?", grade

    if qtype == "grammar_list":
        grades = sorted(chunks_df.loc[chunks_df["doc_type"] == "grammar_concept", "grade"].dropna().unique())
        grade = st.selectbox("2) Grade", [int(g) for g in grades])
        return f"What grammar topics are taught in grade {grade}?", grade

    if qtype == "book_list":
        grades = sorted(chunks_df.loc[chunks_df["doc_type"] == "literature", "grade"].dropna().unique())
        grade = st.selectbox("2) Grade", [int(g) for g in grades])
        return f"What stories or books are assigned in grade {grade}?", grade

    # overview
    grades = sorted(chunks_df.loc[chunks_df["doc_type"] == "grade_overview", "grade"].dropna().unique())
    grade = st.selectbox("2) Grade", [int(g) for g in grades])
    return f"What topics, grammar concepts, and activities are covered in grade {grade}?", grade


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
    st.divider()
    if st.button("🔄 Reset / Clear"):
        st.session_state.clear()
        st.rerun()

if not Path(DOCX_PATH).exists():
    st.error(f"'{DOCX_PATH}' not found next to app.py. Add the curriculum file to the repo.")
    st.stop()

chunks_df, vectorstore, embedding_model = build_index(DOCX_PATH)

mode = st.radio("Question mode", ["Pick from lists (guided)", "Write a free-text question"], horizontal=True)

guided_grade = None
if mode == "Pick from lists (guided)":
    question, guided_grade = build_guided_question(chunks_df)
    st.text_input("Question that will be sent", value=question, disabled=True)
else:
    question = st.text_input("Your question", placeholder="e.g. How do I teach present perfect to grade 5 students?")

ask = st.button("Ask", type="primary")

# In guided mode, the grade picked while building the question takes
# priority over the sidebar filter (it's more specific to this question).
effective_grade = guided_grade if mode == "Pick from lists (guided)" and guided_grade is not None else grade_value

if ask and question.strip():
    with st.spinner("🔍 Searching the curriculum database..."):
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
        with st.expander("📚 View Retrieved Sources"):
            for idx, row in selected_df.iterrows():
                st.markdown(f"**{row['title']}**  ·  score={row['score']:.3f}")
                st.write(row["chunk_text"])
                if idx != selected_df.index[-1]:
                    st.divider()
    else:
        st.info("No matching chunks were retrieved for this question.")
elif ask:
    st.warning("Type a question first.")
