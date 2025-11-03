#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$ROOT_DIR/rendered"

mkdir -p "$OUTPUT_DIR"

if ! command -v dot >/dev/null 2>&1; then
  echo "Graphviz 'dot' binary is required but was not found in PATH." >&2
  exit 1
fi

render_dir() {
  local source_dir="$1"
  local relative="$2"
  if [ ! -d "$source_dir" ]; then
    return
  fi

  find "$source_dir" -maxdepth 1 -type f -name '*.dot' | while read -r dot_file; do
    local base="$(basename "$dot_file" .dot)"
    local target_dir="$OUTPUT_DIR/$relative"
    mkdir -p "$target_dir"
    dot -Tsvg "$dot_file" -o "$target_dir/$base.svg"
    echo "Rendered $relative/$base.svg"
  done
}

render_dir "$ROOT_DIR/skills" "skills"
render_dir "$ROOT_DIR/agent-mappings" "agent-mappings"

cat <<REPORT

============================
Graphviz Validation Complete
============================
Source: $ROOT_DIR
Output: $OUTPUT_DIR
REPORT
