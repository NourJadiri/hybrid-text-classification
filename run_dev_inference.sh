#!/usr/bin/env bash
# Temporary script to run simple_inference.py on five EN dev files
# Usage: chmod +x run_dev_inference.sh && ./run_dev_inference.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT_DIR/.venv"
PY="python"

if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
else
  echo "Warning: virtualenv not found at $VENV. Proceeding with system python."
fi

MODEL_PATH="models/qwen-1.7b-finetuned/final_model"
BASE_MODEL="unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
SCRIPT="src/scripts/simple_inference.py"

# Select first 10 files from devset/EN/subtask-2-documents
TARGET_DIR="$ROOT_DIR/devset/EN/subtask-2-documents"
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: directory not found: $TARGET_DIR" >&2
    exit 1
fi

# Read first 10 regular files (sorted) into FILES array
mapfile -t FILES < <(find "$TARGET_DIR" -maxdepth 1 -type f -print | sort | head -n 5)

if [ "${#FILES[@]}" -eq 0 ]; then
    echo "No files found in $TARGET_DIR" >&2
    exit 1
fi

# Temp file to collect summaries
SUMMARY_TMP="$(mktemp)"

for f in "${FILES[@]}"; do
  echo
  echo "========================"
  echo "FILE: $f"
  echo "------------------------"

  # Run inference and capture output
  OUT="$($PY "$SCRIPT" --file "$f" --model_path "$MODEL_PATH" --base_model "$BASE_MODEL" 2>&1)" || OUT="$OUT"

  # Print full output for debugging
  echo "$OUT"

  # Extract the Detected Narratives line
  NARRATIVES_LINE=$(echo "$OUT" | grep -m1 "Detected Narratives:" || true)
  if [ -z "$NARRATIVES_LINE" ]; then
    # Fallback: try to extract a line that starts with 'Raw Response:' and parse bracketed content
    NARRATIVES_LINE=$(echo "$OUT" | grep -m1 "Raw Response:" || true)
  fi

  # Save to summary tmp (file path -> narratives)
  echo -e "$f	${NARRATIVES_LINE:-<no-detection>}" >> "$SUMMARY_TMP"
done

echo
echo "Batch inference finished. Summary:"
echo "FILE	DETECTED_NARRATIVES"
echo "----	-------------------"
cat "$SUMMARY_TMP"

# Cleanup
rm -f "$SUMMARY_TMP"
