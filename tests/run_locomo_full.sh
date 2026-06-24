#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Full LoCoMo Benchmark Runner
#
# Usage:
#   bash tests/run_locomo_full.sh                    # default: 5 runs × hybrid,agentic
#   bash tests/run_locomo_full.sh --runs 3           # 3 runs
#   bash tests/run_locomo_full.sh --methods hybrid   # hybrid only
#   bash tests/run_locomo_full.sh --skip-add         # reuse existing data
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

RUNS="${RUNS:-5}"
METHODS="${METHODS:-hybrid,agentic}"
SKIP_ADD=""
DATA_PATH="data/locomo10.json"
CONVS=10
POST_FLUSH_WAIT=180
JUDGE_MODEL="gpt-4o-mini"
JUDGE_RUNS=5
TOP_K=10
OUTPUT_DIR="benchmark_results"
SEARCH_CONCURRENCY=1

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs) RUNS="$2"; shift 2 ;;
    --methods) METHODS="$2"; shift 2 ;;
    --skip-add) SKIP_ADD="--skip-add"; shift ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --post-flush-wait) POST_FLUSH_WAIT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="${OUTPUT_DIR}/run_${TS}"
mkdir -p "$RUN_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  LoCoMo Full Benchmark"
echo "  Runs: $RUNS | Methods: $METHODS | Convs: $CONVS"
echo "  Judge: $JUDGE_MODEL (${JUDGE_RUNS} runs/question)"
echo "  Output: $RUN_DIR"
echo "════════════════════════════════════════════════════════════════"

# Phase 1: Add all 10 conversations (once)
if [[ -z "$SKIP_ADD" ]]; then
  echo ""
  echo "──── Phase 1: Loading all $CONVS conversations ────"
  for conv in $(seq 0 $((CONVS - 1))); do
    echo "  Loading conv $conv..."
    uv run python tests/test_locomo.py \
      --conv-index "$conv" \
      --methods hybrid \
      --data-path "$DATA_PATH" \
      --post-flush-wait "$POST_FLUSH_WAIT" \
      --judge-model "$JUDGE_MODEL" \
      --judge-runs 1 \
      --top-k 1 \
      --quiet \
      --search-concurrency 1 \
      --checkpoint-dir "$RUN_DIR/load_conv${conv}" \
      2>&1 | tail -5
    echo "  conv $conv loaded."
  done
  echo "  All conversations loaded."
  SKIP_ADD="--skip-add"
fi

# Phase 2: Run benchmark (search + answer + judge)
IFS=',' read -ra METHOD_LIST <<< "$METHODS"

for run_idx in $(seq 1 "$RUNS"); do
  for method in "${METHOD_LIST[@]}"; do
    echo ""
    echo "══════════════════════════════════════════════════════════"
    echo "  Run $run_idx/$RUNS — method=$method"
    echo "══════════════════════════════════════════════════════════"

    run_out="$RUN_DIR/${method}_run${run_idx}"
    mkdir -p "$run_out"
    summary_file="$run_out/summary.json"

    all_correct=0
    all_total=0

    for conv in $(seq 0 $((CONVS - 1))); do
      conv_out="$run_out/conv${conv}"
      result_file="$conv_out/${method}_results.json"

      echo "  conv $conv ($method, run $run_idx)..."
      uv run python tests/test_locomo.py \
        --conv-index "$conv" \
        --methods "$method" \
        --data-path "$DATA_PATH" \
        --skip-add \
        --judge-model "$JUDGE_MODEL" \
        --judge-runs "$JUDGE_RUNS" \
        --top-k "$TOP_K" \
        --quiet \
        --search-concurrency "$SEARCH_CONCURRENCY" \
        --checkpoint-dir "$conv_out" \
        --output "$result_file" \
        2>&1 | grep -E "Overall:|Done:" | head -3

      # Extract accuracy from result JSON
      if [[ -f "$result_file" ]]; then
        conv_correct=$(python3 -c "
import json, sys
d = json.load(open('$result_file'))
s = d.get('methods', {}).get('$method', {}).get('summary', {})
print(s.get('correct', 0))
" 2>/dev/null || echo 0)
        conv_total=$(python3 -c "
import json, sys
d = json.load(open('$result_file'))
s = d.get('methods', {}).get('$method', {}).get('summary', {})
print(s.get('total', 0))
" 2>/dev/null || echo 0)
        all_correct=$((all_correct + conv_correct))
        all_total=$((all_total + conv_total))
      fi
    done

    # Write run summary
    if [[ $all_total -gt 0 ]]; then
      accuracy=$(python3 -c "print(f'{$all_correct / $all_total * 100:.1f}')")
    else
      accuracy="0.0"
    fi
    echo "{\"method\": \"$method\", \"run\": $run_idx, \"correct\": $all_correct, \"total\": $all_total, \"accuracy\": $accuracy}" > "$summary_file"
    echo "  ── Run $run_idx $method: $all_correct / $all_total ($accuracy%) ──"
  done
done

# Final summary
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Final Results"
echo "════════════════════════════════════════════════════════════════"
for method in "${METHOD_LIST[@]}"; do
  echo "  $method:"
  for run_idx in $(seq 1 "$RUNS"); do
    summary="$RUN_DIR/${method}_run${run_idx}/summary.json"
    if [[ -f "$summary" ]]; then
      python3 -c "
import json
d = json.load(open('$summary'))
print(f\"    Run {d['run']}: {d['correct']}/{d['total']} ({d['accuracy']}%)\")
"
    fi
  done
done
echo "════════════════════════════════════════════════════════════════"
