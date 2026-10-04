"""Assembles the evidence section of the answer prompt."""

# Keep prompts short: only the strongest passage goes to the model.
MAX_EVIDENCE = 1


def build_evidence(docs: list[dict]) -> list[dict]:
    """Select which retrieved documents are passed to answer generation."""
    return docs[:MAX_EVIDENCE]


def build_prompt(system: str, question: str, docs: list[dict]) -> str:
    evidence = build_evidence(docs)
    lines = "\n".join(f"[{d['id']}] {d['text']}" for d in evidence)
    return f"{system}\n\nEvidence:\n{lines}\n\nQuestion: {question}"
