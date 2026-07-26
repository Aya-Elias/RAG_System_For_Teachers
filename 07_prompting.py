"""
07_prompting.py — Stage 7 of the RAG pipeline: builds the grounded
prompt from retrieved context, calls the LLM to generate an answer, and
cleans up the raw model output. Also holds the worksheet/quiz prompt
builder used by the worksheet generator feature in streamlit_app.py.

Generation goes through the HuggingFace Inference API
(huggingface_hub InferenceClient) rather than local model weights, to
keep RAM/CPU usage low enough for Streamlit Community Cloud's free
tier. The token is passed in by the caller (streamlit_app.py reads it
from st.secrets) — this file never reads secrets directly.

Run directly for a quick smoke test (needs an HF_TOKEN env var):
    python 07_prompting.py
"""

import os
import re

from huggingface_hub import InferenceClient

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
            "[No LLM response — no HuggingFace token was configured. "
            "Ask the app owner to add HF_TOKEN in Streamlit secrets.]",
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


WORKSHEET_PROMPT_TEMPLATE = """You are an English Curriculum Assistant creating classroom materials for a teacher.

Using ONLY the grammar concept information below, create a {kind} for grade {grade} students on "{topic}".

Grammar concept information:
{context}

Requirements:
- Write exactly {num_questions} questions, numbered 1 to {num_questions}.
- Base every question strictly on the scope, vocabulary, and examples given above. Do not introduce grammar points that are not mentioned in the information above.
- Use simple, age-appropriate language for grade {grade} students.
- Vary the question types where appropriate (fill-in-the-blank, multiple choice, short answer).
{answer_key_instruction}

Output format:
- A short title line
- The numbered questions
{answer_key_label}"""


def build_worksheet_prompt(topic_text, topic, grade, kind, num_questions):
    is_quiz = kind == "quiz"
    answer_key_instruction = (
        "- After the questions, include an answer key."
        if is_quiz
        else "- Do NOT include an answer key — this is for independent student practice."
    )
    answer_key_label = "- An 'Answer Key' section at the end" if is_quiz else ""
    return WORKSHEET_PROMPT_TEMPLATE.format(
        kind="quiz (with an answer key)" if is_quiz else "practice worksheet (no answer key)",
        grade=grade,
        topic=topic,
        context=topic_text,
        num_questions=num_questions,
        answer_key_instruction=answer_key_instruction,
        answer_key_label=answer_key_label,
    )


if __name__ == "__main__":
    prompt = build_grounded_prompt("What is present perfect?", "Present perfect describes past actions with present relevance.")
    print(prompt)
    token = os.environ.get("HF_TOKEN", "")
    if token:
        answer, model_used = generate_answer(prompt, token)
        print(f"\n--- Answer (via {model_used}) ---\n{answer}")
    else:
        print("\n(Set HF_TOKEN to also test a live generate_answer() call.)")
