#!/usr/bin/env python3
"""
Check that all .rst files under doc/sphinx/src are reachable from the
toctree starting at doc/sphinx/index.rst.

Exits with non-zero status and lists any missing files.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = REPO_ROOT / "doc" / "sphinx"
INDEX_RST = DOC_ROOT / "index.rst"
SRC_DIR = DOC_ROOT / "src"


def resolve_rst(entry: str, base_dir: Path) -> Path:
    p = Path(entry.strip())
    if not p.suffix:
        p = p.with_suffix(".rst")
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    return p


def parse_toctree_targets(rst_path: Path) -> set[Path]:
    targets: set[Path] = set()
    try:
        text = rst_path.read_text(encoding="utf-8")
    except Exception:
        return targets

    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        raw = lines[i]
        stripped = raw.lstrip()
        if stripped.startswith(".. toctree::"):
            base_indent = len(raw) - len(stripped)
            i += 1
            # Consume option lines and collect entries indented more than base_indent
            while i < n:
                line = lines[i]
                if not line.strip():
                    i += 1
                    continue
                indent = len(line) - len(line.lstrip())
                if indent <= base_indent:
                    break
                content = line.strip()
                # Skip option lines like :maxdepth: and comments
                if content.startswith(":") or content.startswith(".. "):
                    i += 1
                    continue
                # Allow "Title <path>" syntax; extract path inside <>
                if "<" in content and ">" in content:
                    entry = content.split("<")[-1].split(">")[0].strip()
                else:
                    # Strip trailing inline comments
                    entry = content.split(" #", 1)[0].strip()
                if entry and not entry.startswith("http"):
                    target = resolve_rst(entry, rst_path.parent)
                    targets.add(target)
                i += 1
            continue  # handled block; do not fall through to i += 1 below
        i += 1

    return targets


def collect_reachable_from_index(index_path: Path) -> set[Path]:
    reachable: set[Path] = set()
    visited: set[Path] = set()
    queue: list[Path] = [index_path]

    while queue:
        current = queue.pop(0)
        if current in visited or not current.exists():
            continue
        visited.add(current)
        for target in parse_toctree_targets(current):
            # Only follow .rst files
            if target.suffix == ".rst" and target.exists():
                if target not in reachable:
                    reachable.add(target)
                    queue.append(target)
    return reachable


def main() -> int:
    if not INDEX_RST.exists():
        print(f"ERROR: index.rst not found at {INDEX_RST}", file=sys.stderr)
        return 2
    if not SRC_DIR.exists():
        print(f"ERROR: docs source directory not found at {SRC_DIR}", file=sys.stderr)
        return 2

    all_rst = {p.resolve() for p in SRC_DIR.rglob("*.rst")}

    reachable = collect_reachable_from_index(INDEX_RST)

    missing = sorted(p for p in all_rst if p not in reachable)

    if missing:
        print("RST files not referenced by toctree starting at index.rst:")
        for p in missing:
            try:
                rel = p.relative_to(REPO_ROOT)
            except ValueError:
                rel = p
            print(f" - {rel}")
        return 1

    print("All RST files under doc/sphinx/src are referenced in the toctree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

