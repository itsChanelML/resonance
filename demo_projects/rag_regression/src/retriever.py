"""Lexical retriever: ranks corpus documents by shared-word count."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STOPWORDS = {"the", "a", "an", "is", "are", "to", "of", "for", "on", "my", "can", "be", "or", "how", "from", "within"}


def load_config(path: Path = ROOT / "config" / "retrieval.yaml") -> dict:
    """Parses the flat `key: value` lines we use; no YAML dependency."""
    config = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if ":" in line:
            key, value = line.split(":", 1)
            config[key.strip()] = int(value) if value.strip().isdigit() else value.strip()
    return config


def load_corpus(path: Path = ROOT / "eval" / "corpus.jsonl") -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS}


def retrieve(question: str, corpus: list[dict], top_k: int) -> list[dict]:
    """Return the top_k documents, best match first (ties keep corpus order)."""
    q = _tokens(question)
    ranked = sorted(corpus, key=lambda d: -len(q & _tokens(d["text"])))
    return ranked[:top_k]
