"""Validated read-only tools the model may call: list_files, search_code,
read_file, git_diff (brief B04, B05, B11). All paths go through
ProjectSession so exclusions and the root boundary apply to every tool.
Arguments are type-checked and bounded; git runs with a fixed argv, never a
model-built shell string."""

import fnmatch
import hashlib
import os
import re
import subprocess
import time
from pathlib import Path

from shared.project_context import (
    BLOCKED, ERROR, NOT_FOUND, NOT_GIT, OK, SUPPORTED_SUFFIXES, TIMEOUT, TOO_LARGE,
    UNSUPPORTED, ContextSource, ProjectSession, ToolResult, redact,
)

MAX_LIST_ENTRIES = 200
MAX_READ_LINES = 200
MAX_SEARCH_MATCHES = 20
MAX_SEARCH_FILES = 2000
MAX_DIFF_CHARS = 12_000
MAX_SEARCH_FILE_BYTES = 256_000
GIT_TIMEOUT_S = 10
_REV = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./~^@-]{0,79}$")  # no leading '-', so never an option

SCHEMAS = [
    {"type": "function", "function": {
        "name": "list_files",
        "description": "List files under a project directory (relative path). Excluded files are never shown.",
        "parameters": {"type": "object", "properties": {
            "relative_path": {"type": "string", "description": "directory, default '.'"},
            "max_entries": {"type": "integer", "description": f"1-{MAX_LIST_ENTRIES}"}},
            "required": []}}},
    {"type": "function", "function": {
        "name": "search_code",
        "description": "Literal (not regex) case-sensitive text search across project files. Use for function names, symbols, config keys.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "glob": {"type": "string", "description": "filename pattern such as *.py, default *"},
            "max_matches": {"type": "integer", "description": f"1-{MAX_SEARCH_MATCHES}"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_file",
        "description": f"Read lines of a project file (max {MAX_READ_LINES} lines per call). Line numbers are 1-based and inclusive.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer"},
            "end_line": {"type": "integer"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "git_diff",
        "description": "Show uncommitted changes. target: unstaged, staged, or baseline (compare working tree to a commit/branch given in 'baseline').",
        "parameters": {"type": "object", "properties": {
            "target": {"type": "string", "enum": ["unstaged", "staged", "baseline"]},
            "baseline": {"type": "string", "description": "commit or branch, only for target=baseline"}},
            "required": ["target"]}}},
]


def _int(args: dict, key: str, default: int, lo: int, hi: int) -> int:
    value = args.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return max(lo, min(hi, value))


def _str(args: dict, key: str, default: str | None = None) -> str:
    value = args.get(key, default)
    if not isinstance(value, str) or (not value and key != "relative_path"):
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _source(path: str, start: int, end: int, content: str, provenance: str) -> ContextSource:
    return ContextSource(
        path=path, start_line=start, end_line=end, content=content,
        content_hash=hashlib.sha256(content.encode()).hexdigest()[:12],
        captured_at=time.time(), provenance=provenance,
    )


class ProjectTools:
    def __init__(self, session: ProjectSession):
        self.session = session

    schemas = SCHEMAS

    def call(self, name: str, args) -> ToolResult:
        """Validate and run one tool. Never raises; bad input becomes ERROR."""
        handlers = {
            "list_files": self.list_files, "search_code": self.search_code,
            "read_file": self.read_file, "git_diff": self.git_diff,
        }
        if name not in handlers:
            return ToolResult(ERROR, f"unknown tool {name!r}", output=f"ERROR: unknown tool {name!r}")
        if not isinstance(args, dict):
            return ToolResult(ERROR, "arguments must be an object", output="ERROR: arguments must be a JSON object")
        try:
            result = handlers[name](args)
        except ValueError as exc:
            result = ToolResult(ERROR, str(exc))
        except Exception as exc:  # a tool bug must not kill the investigation
            result = ToolResult(ERROR, f"{type(exc).__name__}: {exc}")
        if not result.output:
            result.output = f"{result.status}: {result.message}"
        return result

    # -- traversal shared by list_files and search_code -------------------
    def _walk(self, start: Path):
        """Yield (rel_path, abs_path) for readable, non-excluded files under
        start, depth-first and sorted. Excluded dirs are not descended and
        symlinks that leave the root are skipped."""
        root = self.session.root
        stack = [start]
        while stack:
            directory = stack.pop()
            try:
                entries = sorted(os.scandir(directory), key=lambda e: e.name, reverse=True)
            except OSError:
                continue
            for entry in entries:
                try:
                    resolved = Path(entry.path).resolve()
                    rel = resolved.relative_to(root)
                except (ValueError, OSError, RuntimeError):
                    continue
                if self.session.is_excluded(rel):
                    continue
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                elif entry.is_file():
                    yield Path(entry.path).relative_to(root), resolved

    def list_files(self, args: dict) -> ToolResult:
        rel_dir = _str(args, "relative_path", ".")
        limit = _int(args, "max_entries", 100, 1, MAX_LIST_ENTRIES)
        target = (self.session.root / (rel_dir or ".")).resolve()
        try:
            rel = target.relative_to(self.session.root)
        except ValueError:
            return ToolResult(BLOCKED, f"{rel_dir}: outside project root")
        if rel != Path(".") and self.session.is_excluded(rel):
            return ToolResult(BLOCKED, f"{rel}: excluded by policy")
        if not target.is_dir():
            return ToolResult(NOT_FOUND, f"{rel_dir}: not a directory")
        lines, truncated = [], False
        for count, (rel_path, abs_path) in enumerate(self._walk(target)):
            if count >= limit:
                truncated = True
                break
            lines.append(f"{rel_path.as_posix()}  ({abs_path.stat().st_size} bytes)")
        note = f"\n(truncated at {limit} entries)" if truncated else ""
        return ToolResult(OK, output=("\n".join(lines) or "(no files)") + note, truncated=truncated)

    def search_code(self, args: dict) -> ToolResult:
        query = _str(args, "query")
        if len(query) > 200:
            raise ValueError("query too long (max 200 chars)")
        pattern = _str(args, "glob", "*")
        limit = _int(args, "max_matches", MAX_SEARCH_MATCHES, 1, MAX_SEARCH_MATCHES)
        matches, searched, without, sources = [], 0, [], []
        truncated = False
        for rel_path, abs_path in self._walk(self.session.root):
            rel_str = rel_path.as_posix()
            if not (fnmatch.fnmatch(rel_str, pattern) or fnmatch.fnmatch(rel_path.name, pattern)):
                continue
            if abs_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                if abs_path.stat().st_size > MAX_SEARCH_FILE_BYTES:
                    continue
                raw = abs_path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw[:4096]:
                continue
            if searched >= MAX_SEARCH_FILES:
                truncated = True
                break
            searched += 1
            hit = False
            for n, line in enumerate(raw.decode("utf-8", errors="replace").splitlines(), 1):
                if query in line:
                    hit = True
                    if len(matches) >= limit:
                        truncated = True
                        break
                    text, _ = redact(line.strip()[:200])
                    matches.append(f"{rel_str}:{n}: {text}")
                    sources.append(_source(rel_str, n, n, text, "search"))
            if not hit:
                without.append(rel_str)
            if truncated and len(matches) >= limit:
                break
        if not matches:
            return ToolResult(OK, output=f"0 matches for {query!r} in {searched} files searched.")
        tail = f"\n(stopped at {limit} matches; more exist)" if truncated else ""
        none_note = ""
        if without:
            none_note = "\nSearched, no match: " + ", ".join(without[:15]) + (" ..." if len(without) > 15 else "")
        return ToolResult(OK, sources=sources, truncated=truncated,
                          output="\n".join(matches) + tail + none_note)

    def read_file(self, args: dict) -> ToolResult:
        path = _str(args, "path")
        start = _int(args, "start_line", 1, 1, 10**9)
        end = _int(args, "end_line", start + MAX_READ_LINES - 1, start, 10**9)
        end = min(end, start + MAX_READ_LINES - 1)
        result = self.session.read_file(path, start, end, provenance="tool")
        if result.status != OK:
            return result
        src = result.sources[0]
        numbered = "\n".join(f"{n}: {line}" for n, line in enumerate(src.content.splitlines(), src.start_line))
        return ToolResult(OK, sources=[src], truncated=src.truncated,
                          output=f"{src.ref}\n{numbered}" if numbered else f"{src.path}: no lines in range {start}-{end}")

    # -- git ---------------------------------------------------------------
    def _git(self, *argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.session.root), *argv],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S, check=False,
        )

    def git_diff(self, args: dict) -> ToolResult:
        target = _str(args, "target")
        if target not in ("unstaged", "staged", "baseline"):
            raise ValueError("target must be unstaged, staged, or baseline")
        diff_argv = ["diff", "--no-color", "--relative"]
        if target == "staged":
            diff_argv.append("--cached")
        elif target == "baseline":
            baseline = _str(args, "baseline")
            if not _REV.match(baseline) or ".." in baseline:
                raise ValueError("baseline must be a simple commit or branch name")
            diff_argv.append(baseline)
        try:
            inside = self._git("rev-parse", "--is-inside-work-tree")
            if inside.returncode != 0:
                return ToolResult(NOT_GIT, "project is not a git repository")
            proc = self._git(*diff_argv, "--", ".")
            untracked = self._git("ls-files", "--others", "--exclude-standard")
        except FileNotFoundError:
            return ToolResult(NOT_GIT, "git is not installed")
        except subprocess.TimeoutExpired:
            return ToolResult(TIMEOUT, f"git timed out after {GIT_TIMEOUT_S}s")
        if proc.returncode != 0:
            return ToolResult(ERROR, proc.stderr.strip()[:200] or "git diff failed")

        blocks = re.split(r"(?m)^(?=diff --git )", proc.stdout)
        kept, sources, blocked = [], [], 0
        for block in blocks:
            if not block.strip():
                continue
            m = re.match(r"diff --git a/(.*?) b/(.*)", block)
            path = m.group(2) if m else ""
            if self.session.is_excluded(Path(path)):
                blocked += 1
                continue
            clean, _ = redact(block)
            kept.append(clean)
            for h in re.finditer(r"(?m)^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", clean):
                start, length = int(h.group(1)), int(h.group(2) or 1)
                sources.append(_source(path, start, start + max(length, 1) - 1, clean, "git_diff"))
        text = "".join(kept)
        truncated = len(text) > MAX_DIFF_CHARS
        text = text[:MAX_DIFF_CHARS]
        notes = []
        if truncated:
            notes.append(f"(diff truncated at {MAX_DIFF_CHARS} chars)")
        if blocked:
            notes.append(f"({blocked} changed file(s) hidden by exclusion policy)")
        new_files = [f for f in untracked.stdout.splitlines()
                     if f and not self.session.is_excluded(Path(f))]
        if new_files:
            notes.append("Untracked files (not in diff): " + ", ".join(new_files[:20]))
        if not text and not notes:
            return ToolResult(OK, output=f"No {target} changes (clean).")
        return ToolResult(OK, sources=sources, truncated=truncated,
                          output="\n".join(filter(None, [text, *notes])))
