#!/usr/bin/env python3
"""Find and optionally remove repository clutter before open-source handoff."""

from __future__ import annotations

import argparse
import bisect
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

DIR_REASONS: dict[str, str] = {
    "__pycache__": "Python bytecode cache directory",
    ".pytest_cache": "pytest cache directory",
    ".ruff_cache": "ruff cache directory",
    ".mypy_cache": "mypy cache directory",
    ".tox": "tox virtualenv output",
    ".nox": "nox virtualenv output",
    "coverage": "coverage output directory",
    "htmlcov": "HTML coverage report directory",
    "dist": "build artifact directory",
    "build": "build artifact directory",
    ".next": "Next.js build output",
    ".svelte-kit": "SvelteKit build output",
    ".parcel-cache": "Parcel cache directory",
    ".turbo": "Turborepo cache directory",
    ".cache": "local cache directory",
    ".ipynb_checkpoints": "Jupyter checkpoint directory",
    "node_modules": "local dependency installation directory",
    ".venv": "local Python virtual environment directory",
    "venv": "local Python virtual environment directory",
    "target": "Rust build artifact directory",
    ".idea": "JetBrains project metadata",
}

FILE_REASONS: dict[str, str] = {
    ".DS_Store": "macOS Finder metadata file",
    "Thumbs.db": "Windows Explorer metadata file",
    ".coverage": "coverage database file",
}

FILE_SUFFIX_REASONS: dict[str, str] = {
    ".pyc": "Python bytecode artifact",
    ".pyo": "Python optimized bytecode artifact",
    ".tmp": "temporary file",
    ".log": "log artifact",
}


@dataclass
class Candidate:
    path: str
    kind: str
    reason: str
    git_status: str


def git_is_repo(root: Path) -> bool:
    try:
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--is-inside-work-tree"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=False,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def git_path_set(root: Path, args: list[str]) -> set[str]:
    try:
        proc = subprocess.run(
            ["git", "-C", str(root), "ls-files", *args, "-z"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=False,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return set()
    return {
        chunk.decode("utf-8", errors="replace")
        for chunk in proc.stdout.split(b"\x00")
        if chunk
    }


def has_prefix(paths_sorted: list[str], prefix: str) -> bool:
    idx = bisect.bisect_left(paths_sorted, prefix)
    return idx < len(paths_sorted) and paths_sorted[idx].startswith(prefix)


def reason_for_directory(name: str) -> str | None:
    return DIR_REASONS.get(name)


def reason_for_file(name: str) -> str | None:
    if name in FILE_REASONS:
        return FILE_REASONS[name]
    for suffix, reason in FILE_SUFFIX_REASONS.items():
        if name.endswith(suffix):
            return reason
    return None


def discover_candidates(root: Path) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for current, dirnames, filenames in os.walk(root, topdown=True):
        current_path = Path(current)
        rel_parent = "" if current_path == root else current_path.relative_to(root).as_posix()

        dirnames[:] = [d for d in dirnames if d != ".git"]

        keep_dirs: list[str] = []
        for dirname in dirnames:
            rel_path = f"{rel_parent}/{dirname}" if rel_parent else dirname
            reason = reason_for_directory(dirname)
            if reason:
                found.append((rel_path, "dir", reason))
            else:
                keep_dirs.append(dirname)
        dirnames[:] = keep_dirs

        for filename in filenames:
            reason = reason_for_file(filename)
            if not reason:
                continue
            rel_path = f"{rel_parent}/{filename}" if rel_parent else filename
            found.append((rel_path, "file", reason))
    return found


def classify_status(
    path: str,
    kind: str,
    tracked: set[str],
    untracked: set[str],
    ignored: set[str],
    tracked_sorted: list[str],
    untracked_sorted: list[str],
    ignored_sorted: list[str],
) -> str:
    if kind == "file":
        if path in tracked:
            return "tracked"
        if path in untracked:
            return "untracked"
        if path in ignored:
            return "ignored"
        return "unknown"

    prefix = f"{path}/"
    if path in tracked or has_prefix(tracked_sorted, prefix):
        return "tracked"
    if path in untracked or has_prefix(untracked_sorted, prefix):
        return "untracked"
    if path in ignored or has_prefix(ignored_sorted, prefix):
        return "ignored"
    return "unknown"


def summarize(candidates: list[Candidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in candidates:
        counts[item.git_status] = counts.get(item.git_status, 0) + 1
    return counts


def should_delete(candidate: Candidate, include_tracked: bool) -> bool:
    if candidate.git_status in {"untracked", "ignored"}:
        return True
    if candidate.git_status == "tracked" and include_tracked:
        return True
    return False


def write_report(path: Path, root: Path, candidates: list[Candidate], deleted: list[str]) -> None:
    counts = summarize(candidates)
    lines = [
        "# Repository Cleanup Report",
        "",
        f"- Root: `{root}`",
        f"- Candidates: **{len(candidates)}**",
        f"- Deleted: **{len(deleted)}**",
        "",
        "## Candidate Status Counts",
    ]
    for key in sorted(counts):
        lines.append(f"- {key}: {counts[key]}")

    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| Path | Kind | Git Status | Reason |",
            "|---|---|---|---|",
        ]
    )
    for item in sorted(candidates, key=lambda c: c.path):
        lines.append(
            f"| `{item.path}` | {item.kind} | {item.git_status} | {item.reason.replace('|', '/')} |"
        )

    if deleted:
        lines.extend(["", "## Deleted", ""])
        for entry in deleted:
            lines.append(f"- `{entry}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan for repository clutter and optionally delete safe candidates."
    )
    parser.add_argument("--root", default=".", help="Repository root to scan (default: current directory)")
    parser.add_argument("--apply", action="store_true", help="Delete matched paths")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Confirm deletion when --apply is used",
    )
    parser.add_argument(
        "--include-tracked",
        action="store_true",
        help="Also delete tracked candidates (dangerous; use only with explicit approval)",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--report", type=Path, help="Write a markdown report to this path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()

    if not root.exists() or not root.is_dir():
        print(f"error: root directory not found: {root}", file=sys.stderr)
        return 2

    candidates_raw = discover_candidates(root)

    if git_is_repo(root):
        tracked = git_path_set(root, [])
        untracked = git_path_set(root, ["--others", "--exclude-standard"])
        ignored = git_path_set(root, ["--others", "--ignored", "--exclude-standard"])
        tracked_sorted = sorted(tracked)
        untracked_sorted = sorted(untracked)
        ignored_sorted = sorted(ignored)
    else:
        tracked = set()
        untracked = set()
        ignored = set()
        tracked_sorted = []
        untracked_sorted = []
        ignored_sorted = []

    candidates = [
        Candidate(
            path=path,
            kind=kind,
            reason=reason,
            git_status=classify_status(
                path,
                kind,
                tracked,
                untracked,
                ignored,
                tracked_sorted,
                untracked_sorted,
                ignored_sorted,
            ),
        )
        for path, kind, reason in candidates_raw
    ]

    deleted: list[str] = []
    if args.apply:
        if not args.yes:
            print("error: --apply requires --yes", file=sys.stderr)
            return 2

        for candidate in candidates:
            if not should_delete(candidate, args.include_tracked):
                continue
            target = root / candidate.path
            if not target.exists():
                continue
            if candidate.kind == "dir":
                shutil.rmtree(target)
            else:
                target.unlink()
            deleted.append(candidate.path)

    if args.report:
        write_report(args.report.resolve(), root, candidates, deleted)

    if args.json:
        payload = {
            "root": str(root),
            "candidate_count": len(candidates),
            "status_counts": summarize(candidates),
            "deleted_count": len(deleted),
            "candidates": [asdict(c) for c in candidates],
            "deleted": deleted,
        }
        print(json.dumps(payload, indent=2))
        return 0

    counts = summarize(candidates)
    print(f"Root: {root}")
    print(f"Candidates found: {len(candidates)}")
    for key in sorted(counts):
        print(f"- {key}: {counts[key]}")

    if candidates:
        print("\nTop candidates:")
        for item in sorted(candidates, key=lambda c: c.path)[:25]:
            print(f"- [{item.git_status}] {item.path} ({item.reason})")

    if len(candidates) > 25:
        print(f"... and {len(candidates) - 25} more")

    if args.apply:
        print(f"\nDeleted: {len(deleted)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
