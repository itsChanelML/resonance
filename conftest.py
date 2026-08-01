import sys
from pathlib import Path

# So `import shared...` resolves regardless of how pytest is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))
