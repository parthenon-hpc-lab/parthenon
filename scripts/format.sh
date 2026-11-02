#!/usr/bin/env bash
# This file was made in part with generative AI
#
# Formats all tracked C++ and Python files.
#
# By default, the clang-format and black versions pinned in .pre-commit-config.yaml are
# used (installed automatically by pre-commit).
# If pre-commit is not available, or with --local, locally installed tools are used
# instead (can be overridden by the CLANG_FORMAT and BLACK environment variables).
# Note that their results may differ from the CI if their versions differ.

set -euo pipefail

usage() {
  echo "Usage: $0 [--local]"
  echo "  --local  use locally installed clang-format and black instead of pre-commit"
}

use_local=false
case "${1-}" in
  "") ;;
  --local) use_local=true ;;
  -h | --help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 1
    ;;
esac

cd "$(git rev-parse --show-toplevel)"

if ! "${use_local}"; then
  if command -v pre-commit >/dev/null 2>&1; then
    for hook in clang-format black; do
      # pre-commit also returns non-zero if files were modified, so rerun on failure
      # to distinguish modified files from actual errors.
      if ! pre-commit run --all-files "${hook}" >/dev/null 2>&1; then
        pre-commit run --all-files "${hook}"
      fi
    done
    git diff --stat
    exit 0
  fi
  echo "warning: pre-commit not found, using locally installed tools." >&2
fi

# Version pinned in .pre-commit-config.yaml (from the "# frozen: <version>" comment)
pinned_version() {
  awk -v repo="$1" '$0 ~ "repo: .*" repo { found = 1 }
    found && /# frozen:/ { sub(/.*# frozen: v?/, ""); print; exit }' \
    .pre-commit-config.yaml
}

# Warn if the major version of the local tool differs from the pinned one.
check_version() {
  local name="$1" version="$2" pinned="$3"
  echo "Using ${name} ${version} (pinned: ${pinned})"
  if [[ "${version%%.*}" != "${pinned%%.*}" ]]; then
    echo "warning: ${name} ${version} differs from the pinned version ${pinned}." >&2
    echo "warning: The formatting may differ from the one checked in the CI." >&2
  fi
}

status=1 # becomes 0 once at least one tool ran

# C++ files. Keep the extensions in sync with the clang-format hook `types_or`.
clang_format_pinned="$(pinned_version mirrors-clang-format)"
clang_format_major="${clang_format_pinned%%.*}"
clang_format="${CLANG_FORMAT-}"
if [[ -z "${clang_format}" ]]; then
  for candidate in "clang-format-${clang_format_major}" \
    "clang-format-mp-${clang_format_major}" clang-format; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      clang_format="${candidate}"
      break
    fi
  done
fi
if [[ -n "${clang_format}" ]] && command -v "${clang_format}" >/dev/null 2>&1; then
  version="$("${clang_format}" --version |
    sed -n 's/.*clang-format version \([0-9][0-9.]*\).*/\1/p')"
  check_version clang-format "${version}" "${clang_format_pinned}"
  files=()
  while IFS= read -r -d '' file; do
    files+=("${file}")
  done < <(git ls-files -z -- '*.c' '*.h' '*.C' '*.H' '*.cc' '*.hh' '*.cpp' '*.hpp' \
    '*.cxx' '*.hxx' '*.c++' '*.h++')
  if ((${#files[@]} > 0)); then
    "${clang_format}" -style=file -i "${files[@]}"
  fi
  status=0
else
  echo "warning: clang-format not found, skipping C++ files." >&2
fi

# Python files. Keep the extensions in sync with the black hook `types_or`.
black_pinned="$(pinned_version black-pre-commit-mirror)"
black="${BLACK:-black}"
if command -v "${black}" >/dev/null 2>&1; then
  version="$("${black}" --version | sed -n 's/^black, \([0-9][0-9.]*\).*/\1/p')"
  check_version black "${version}" "${black_pinned}"
  files=()
  while IFS= read -r -d '' file; do
    files+=("${file}")
  done < <(git ls-files -z -- '*.py' '*.pyi')
  if ((${#files[@]} > 0)); then
    "${black}" --quiet "${files[@]}"
  fi
  status=0
else
  echo "warning: black not found, skipping Python files." >&2
fi

if ((status != 0)); then
  echo "error: neither clang-format nor black found." >&2
  exit 1
fi

git diff --stat
