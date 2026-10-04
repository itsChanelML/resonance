"""Deterministic evaluation runner. Answer generation is stubbed: an answer is
grounded only if the supporting document reached the prompt, so the score
isolates retrieval and prompt assembly from model behavior.

    python eval/run_eval.py --out artifacts/eval_after.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompt_builder import build_evidence  # noqa: E402
from src.retriever import ROOT, load_config, load_corpus, retrieve  # noqa: E402


def run() -> dict:
    corpus = load_corpus()
    top_k = load_config()["top_k"]
    results = []
    for line in (ROOT / "eval" / "cases.jsonl").read_text().splitlines():
        case = json.loads(line)
        retrieved = retrieve(case["question"], corpus, top_k)
        evidence = build_evidence(retrieved)
        grounded = any(d["id"] == case["support"] for d in evidence)
        results.append({
            "case_id": case["id"],
            "question": case["question"],
            "supporting_doc": case["support"],
            "retrieved_ids": [d["id"] for d in retrieved],
            "evidence_ids": [d["id"] for d in evidence],
            "answer": next(d["text"] for d in evidence if d["id"] == case["support"]) if grounded
            else "I do not know.",
            "scores": {
                "retrieval_recall": int(case["support"] in [d["id"] for d in retrieved]),
                "grounded_accuracy": int(grounded),
            },
        })
    n = len(results)
    return {
        "top_k": top_k,
        "summary": {
            "retrieval_recall": sum(r["scores"]["retrieval_recall"] for r in results) / n,
            "grounded_accuracy": sum(r["scores"]["grounded_accuracy"] for r in results) / n,
        },
        "cases": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = run()
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))
