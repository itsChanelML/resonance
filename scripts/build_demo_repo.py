"""Build a ready-to-demo git repo for the RAG regression scenario.

    python scripts/build_demo_repo.py /tmp/rag_demo
    python resonance.py --project /tmp/rag_demo --context artifacts/eval_after.json

Committed state is the "before" code and eval results. The working tree holds
the uncommitted change (wider retrieval plus the prompt-assembly edit) and
the "after" eval results, so git_diff shows exactly what changed.
"""

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "demo_projects" / "rag_regression"
BASELINE = REPO / "demo_projects" / "_baseline" / "rag_regression"


def git(dest: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(dest), "-c", "user.email=demo@example.com",
                    "-c", "user.name=Demo", *args], check=True, capture_output=True)


def run_eval(dest: Path, out: str) -> None:
    subprocess.run([sys.executable, "eval/run_eval.py", "--out", out],
                   cwd=dest, check=True, capture_output=True)


def main(dest_arg: str) -> None:
    dest = Path(dest_arg).resolve()
    if dest.exists():
        shutil.rmtree(dest)
    ignore = shutil.ignore_patterns("__pycache__", "artifacts")
    shutil.copytree(FIXTURE, dest, ignore=ignore)
    (dest / "artifacts").mkdir()

    # before state: committed
    after_files = {}
    for path in BASELINE.rglob("*"):
        if path.is_file():
            rel = path.relative_to(BASELINE)
            after_files[rel] = (dest / rel).read_bytes()
            (dest / rel).write_bytes(path.read_bytes())
    run_eval(dest, "artifacts/eval_before.json")
    git(dest, "init", "-q")
    git(dest, "add", ".")
    git(dest, "commit", "-qm", "Baseline: top_k=3, evidence=3")

    # after state: uncommitted working-tree change
    for rel, content in after_files.items():
        (dest / rel).write_bytes(content)
    run_eval(dest, "artifacts/eval_after.json")
    print(f"Demo repo ready at {dest}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/rag_demo")
