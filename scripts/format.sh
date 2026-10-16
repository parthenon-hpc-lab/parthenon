#!/usr/bin/env bash
# This file was made in part with generative AI
#
# Formats all tracked C++ and Python files using the clang-format and black versions
# pinned in .pre-commit-config.yaml (installed automatically by pre-commit).

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if ! command -v pre-commit >/dev/null 2>&1; then
  echo "error: pre-commit not found. Install it with 'pip install pre-commit'." >&2
  exit 1
fi

for hook in clang-format black; do
  # pre-commit also returns non-zero if files were modified, so rerun on failure
  # to distinguish modified files from actual errors.
  if ! pre-commit run --all-files "${hook}" >/dev/null 2>&1; then
    pre-commit run --all-files "${hook}"
  fi
done

git diff --stat
