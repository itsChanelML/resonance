"""Read-only project workspace: root selection, access policy, and explicit
context attachments (brief B02, B03, B11-lite, B12 statuses).

Every file read goes through ProjectSession.resolve(), which canonicalizes
the path (following symlinks) and refuses anything outside the project root
or matched by the exclusion policy. Nothing here writes to the project."""

import fnmatch
import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

OK = "OK"
NOT_FOUND = "NOT_FOUND"
BLOCKED = "BLOCKED"
TOO_LARGE = "TOO_LARGE"
UNSUPPORTED = "UNSUPPORTED"
ERROR = "ERROR"

SUPPORTED_SUFFIXES = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java", ".c", ".h",
    ".cpp", ".sh", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".txt", ".log",
    ".json", ".jsonl", ".csv", ".md", ".sql", "",
}

# Matched against every path component and the relative path.
DEFAULT_EXCLUDES = [
    ".env", ".env.*", "*.pem", "*.key", "id_rsa*", "id_ed25519*", "*.p12",
    "credentials*", ".git", "node_modules", "venv", ".venv", "__pycache__",
    "dist", "build", "*.pyc", "*.so", "*.bin", "*.png", "*.jpg", "*.zip",
]

SECRET_PATTERNS = [
    re.compile(r"nvapi-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
]

MAX_FILE_BYTES = 256_000


@dataclass
class ContextSource:
    path: str  # relative to project root
    start_line: int
    end_line: int
    content: str  # raw excerpt, no line-number prefixes
    content_hash: str
    captured_at: float
    provenance: str = "attachment"
    truncated: bool = False
    redactions: int = 0

    @property
    def ref(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


@dataclass
class ToolResult:
    status: str
    message: str = ""
    sources: list = field(default_factory=list)


class ProjectSession:
    def __init__(self, root: str | Path, max_lines: int = 400):
        root_path = Path(root).expanduser()
        if not root_path.is_dir():
            raise NotADirectoryError(f"Project root not found: {root}")
        self.root = root_path.resolve()
        self.max_lines = max_lines
        self.excludes = list(DEFAULT_EXCLUDES) + self._load_ignore_file()
        self.attachments: dict[str, ContextSource] = {}

    def _load_ignore_file(self) -> list[str]:
        patterns = []
        for name in (".gitignore", ".resonanceignore"):
            ignore = self.root / name
            if ignore.is_file():
                for ln in ignore.read_text(errors="ignore").splitlines():
                    ln = ln.strip().lstrip("/")
                    if ln and not ln.startswith(("#", "!")):  # negations unsupported: stay conservative
                        patterns.append(ln)
        return patterns

    def is_excluded(self, rel: Path) -> bool:
        rel_str = rel.as_posix()
        for pattern in self.excludes:
            pattern = pattern.rstrip("/")
            if fnmatch.fnmatch(rel_str, pattern):
                return True
            if any(fnmatch.fnmatch(part, pattern) for part in rel.parts):
                return True
        return False

    def resolve(self, user_path: str) -> tuple[Path | None, ToolResult | None]:
        """Canonicalize user_path against the root. Returns (path, None) when
        readable, or (None, ToolResult) describing why not."""
        candidate = Path(user_path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        try:
            resolved = candidate.resolve()  # follows symlinks
        except (OSError, RuntimeError) as exc:
            return None, ToolResult(ERROR, f"{user_path}: {exc}")
        try:
            rel = resolved.relative_to(self.root)
        except ValueError:
            return None, ToolResult(BLOCKED, f"{user_path}: outside project root")
        if self.is_excluded(rel):
            return None, ToolResult(BLOCKED, f"{rel}: excluded by policy")
        if not resolved.exists():
            return None, ToolResult(NOT_FOUND, f"{user_path}: not found")
        if not resolved.is_file():
            return None, ToolResult(UNSUPPORTED, f"{rel}: not a file")
        return resolved, None

    def read_file(self, user_path: str, start_line: int = 1,
                  end_line: int | None = None, provenance: str = "attachment") -> ToolResult:
        path, failure = self.resolve(user_path)
        if failure:
            return failure
        rel = path.relative_to(self.root)
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            return ToolResult(UNSUPPORTED, f"{rel}: unsupported file type {path.suffix}")
        try:
            size = path.stat().st_size
            if size > MAX_FILE_BYTES * 4:
                return ToolResult(TOO_LARGE, f"{rel}: {size} bytes exceeds limit")
            raw = path.read_bytes()
        except OSError as exc:
            return ToolResult(ERROR, f"{rel}: {exc}")
        if b"\x00" in raw[:4096]:
            return ToolResult(UNSUPPORTED, f"{rel}: binary file")
        lines = raw.decode("utf-8", errors="replace").splitlines()

        start = max(1, start_line)
        last = min(len(lines), end_line if end_line else start + self.max_lines - 1)
        truncated = last - start + 1 > self.max_lines
        last = min(last, start + self.max_lines - 1)
        excerpt = lines[start - 1:last]
        truncated = truncated or (end_line is None and last < len(lines))

        redactions = 0
        cleaned = []
        for line in excerpt:
            for pattern in SECRET_PATTERNS:
                line, n = pattern.subn("[REDACTED]", line)
                redactions += n
            cleaned.append(line)
        content = "\n".join(cleaned)
        source = ContextSource(
            path=rel.as_posix(), start_line=start, end_line=max(start, last),
            content=content,
            content_hash=hashlib.sha256(content.encode()).hexdigest()[:12],
            captured_at=time.time(), provenance=provenance,
            truncated=truncated, redactions=redactions,
        )
        return ToolResult(OK, sources=[source])

    def attach(self, user_path: str) -> ToolResult:
        result = self.read_file(user_path)
        if result.status == OK:
            src = result.sources[0]
            self.attachments[src.path] = src
        return result

    def refresh_stale(self) -> list[str]:
        """Re-read attachments whose file changed since capture; returns the
        relative paths that were refreshed."""
        refreshed = []
        for rel, old in list(self.attachments.items()):
            result = self.read_file(rel, old.start_line, old.end_line)
            if result.status == OK and result.sources[0].content_hash != old.content_hash:
                self.attachments[rel] = result.sources[0]
                refreshed.append(rel)
        return refreshed

    def clear(self) -> None:
        self.attachments.clear()
