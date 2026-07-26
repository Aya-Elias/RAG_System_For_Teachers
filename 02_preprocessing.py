"""
02_preprocessing.py — Stage 2 of the RAG pipeline: text cleaning applied
to each raw document row before chunking (normalizes whitespace, strips
decorative separator lines, tidies punctuation spacing).

Run directly for a quick smoke test:
    python 02_preprocessing.py
"""

import re


def clean_text(text):
    if not isinstance(text, str):
        return text
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[-_=]{3,}", " ", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return text.strip()


if __name__ == "__main__":
    sample = "This   is\n a   messy---string , with  spacing ."
    print(clean_text(sample))
