import sys
from pathlib import Path

# So `import shared...` resolves regardless of how pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# The RAG demo fixture contains a deliberately failing test (it proves the seeded bug).
collect_ignore = ["demo_projects"]
