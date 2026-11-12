#!/usr/bin/env bash
# Fetch a benchmark's data to the path its config reads by default.
#
#   bash benchmarks/download_data.sh locomo   # or longmemeval | evermembench | subtlememory | all
#
# Fetches only the files each adapter reads, straight to where it reads them (no renames
# or moves afterwards); keeps files already present, so a rerun finishes an interrupted
# fetch. Sources follow each upstream's main branch.
set -euo pipefail

EVEROS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$EVEROS"
DATA=benchmarks/data
HF=https://huggingface.co/datasets
# A progress bar for a person at a terminal; quiet-but-failing-loudly for a log.
if [ -t 2 ]; then CURL_PROGRESS=--progress-bar; else CURL_PROGRESS=-sS; fi

fetch() {  # fetch <url> <destination>
  local url=$1 dest=$2
  if [ -s "$dest" ]; then
    echo "  have  $dest"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  echo "  get   $dest"
  curl -fL --retry 3 "$CURL_PROGRESS" -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
}

locomo() {
  fetch https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json \
    "$DATA/locomo10.json"
  # The gold audit behind the report's revised score, pinned to the commit it was read
  # at. Licensed CC BY-NC 4.0, like LoCoMo itself.
  fetch https://raw.githubusercontent.com/dial481/locomo-audit/ded8d0f08903d72b5a5eeda4fee5ced76126e7bc/errors.json \
    "$DATA/locomo-audit-errors.json"
}

longmemeval() {
  fetch "$HF/xiaowu0162/longmemeval/resolve/main/longmemeval_s" \
    "$DATA/longmemeval_s.json"
}

evermembench() {
  local raw="$DATA/raw/EverMemBench-Dynamic" t
  for t in 01 02 03 04 05; do
    fetch "$HF/EverMind-AI/EverMemBench-Dynamic/resolve/main/$t/dialogue.json" \
      "$raw/$t/dialogue.json"
    fetch "$HF/EverMind-AI/EverMemBench-Dynamic/resolve/main/$t/qa_$t.json" \
      "$raw/$t/qa_$t.json"
  done
  # Converted once; the raw release stays because the adapter reads session names from it.
  if [ ! -s "$DATA/evermembench.json" ]; then
    if [ ! -x .venv/bin/python ]; then
      echo ".venv not found -- run \`uv sync\` first, then run this again." >&2
      exit 1
    fi
    # Imported, not `-m`: -m warns that the package __init__ already imported it.
    .venv/bin/python -c "from benchmarks.adapters.evermembench import main; main()"
  fi
}

subtlememory() {
  local i f
  for i in 0 1 2 3 4 5 6 7 8 9; do
    for f in bench_instances.json history_sessions.json; do
      fetch "$HF/Yummytanmo/SubtleMemory/resolve/main/data/subtlememory/persona_$i/$f" \
        "$DATA/subtlememory/persona_$i/$f"
    done
  done
}

case "${1:-}" in
  locomo | longmemeval | evermembench | subtlememory) targets=("$1") ;;
  all) targets=(locomo longmemeval evermembench subtlememory) ;;
  *)
    echo "usage: bash benchmarks/download_data.sh <locomo|longmemeval|evermembench|subtlememory|all>" >&2
    exit 2
    ;;
esac

for target in "${targets[@]}"; do
  echo "$target"
  "$target"
done
echo
echo "Data ready. Next (benchmarks/README.md, \"3. Configure\" and \"4. Run\"):"
echo "  DATASET=${targets[0]} bash benchmarks/reproduce.sh --smoke"
echo "  (only an OpenRouter key? add --methods keyword)"
