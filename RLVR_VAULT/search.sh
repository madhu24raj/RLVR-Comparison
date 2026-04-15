#!/usr/bin/env bash
# Usage:
#   ./search.sh "query"           -- search all wiki pages
#   ./search.sh "query" concepts  -- scoped to wiki/concepts/
#
# Returns file paths and matching lines. Exits 0 on match, 1 on no match (rg default).

set -euo pipefail

QUERY="${1:?Usage: search.sh <query> [subdir]}"
SUBDIR="${2:-}"
WIKI_DIR="$(dirname "$0")/wiki"

if [[ -n "$SUBDIR" ]]; then
  TARGET="$WIKI_DIR/$SUBDIR"
else
  TARGET="$WIKI_DIR"
fi

if [[ ! -d "$TARGET" ]]; then
  echo "Error: directory not found: $TARGET" >&2
  exit 1
fi

# Resolve rg: prefer CLAUDE_CODE_EXECPATH (acts as rg when ARGV0=rg), else system rg
if [[ -x "${CLAUDE_CODE_EXECPATH:-}" ]]; then
  ARGV0=rg "$CLAUDE_CODE_EXECPATH" --color=never --heading --line-number "$QUERY" "$TARGET"
else
  rg --color=never --heading --line-number "$QUERY" "$TARGET"
fi
