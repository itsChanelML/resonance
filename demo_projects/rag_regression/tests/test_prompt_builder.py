import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompt_builder import build_evidence  # noqa: E402


def test_evidence_keeps_supporting_document_even_when_not_first():
    """Ground truth for the demo: the supporting doc (d7) is ranked second,
    so truncating evidence to one item drops it."""
    retrieved = [{"id": "d3", "text": "shipping"}, {"id": "d7", "text": "premium shipping"}]
    assert "d7" in [d["id"] for d in build_evidence(retrieved)]
