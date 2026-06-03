#!/usr/bin/env bash
# Batch driver for LoCoMo benchmark across multiple conversations + methods.
#
# Wraps tests/test_locomo.py in an outer ``--conv-index`` loop. test_locomo.py
# already loops over ``--methods`` internally, so one invocation per
# conversation runs the full method matrix for that conv.
#
# Per-conv outputs (separate JSON + checkpoint dir) live under
# ``benchmark_results/run_<timestamp>/conv<N>.json`` so reports never collide.
# An aggregate accuracy table is printed at the end.
#
# Examples
# ────────
#   # all 10 convs, hybrid only:
#   bash tests/run_locomo_batch.sh --conv-indices 0-9 --methods hybrid
#
#   # 3 specific convs, two methods, skip the ~5min Add phase (corpus already loaded):
#   bash tests/run_locomo_batch.sh \
#     --conv-indices 0,3,7 --methods keyword,hybrid --skip-add
#
#   # one conv, all 4 methods comparison:
#   bash tests/run_locomo_batch.sh --conv-indices 0 --methods keyword,vector,hybrid,agentic

set -euo pipefail

# ── Defaults (override via flags) ─────────────────────────────────────
BASE_URL="${BASE_URL:-http://localhost:8000}"
DATA_PATH="${DATA_PATH:-data/locomo10.json}"
CONV_INDICES="${CONV_INDICES:-0}"
METHODS="${METHODS:-hybrid}"
TOP_K="${TOP_K:-10}"
EVAL_OWNER="${EVAL_OWNER:-speaker_a}"
JUDGE_RUNS="${JUDGE_RUNS:-1}"
SKIP_ADD="false"
OUTPUT_ROOT=""
CONCURRENCY="${CONCURRENCY:-1}"
# Default to polling cascade pending==0 (not fixed sleep). Falls back to
# ~/.everos to match the server's default data root; override via env or
# EVEROS_MEMORY__ROOT (which the server consumes). post-flush-wait becomes
# the MAX wait when corpus-path is set.
CORPUS_PATH="${CORPUS_PATH:-${EVEROS_MEMORY__ROOT:-$HOME/.everos}}"
POST_FLUSH_WAIT="${POST_FLUSH_WAIT:-600}"
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: bash tests/run_locomo_batch.sh [options]

  --conv-indices <spec>   conv list — "0,1,2" | "0-9" | "all"    (default: $CONV_INDICES)
  --methods <list>        comma-separated, e.g. "keyword,hybrid"  (default: $METHODS)
  --base-url <url>        everos server                          (default: $BASE_URL)
  --data-path <file>      LoCoMo dataset path                     (default: $DATA_PATH)
  --top-k <int>           per-question recall depth               (default: $TOP_K)
  --eval-owner <a|b>      speaker_a | speaker_b                   (default: $EVAL_OWNER)
  --judge-runs <int>      LLM judge majority-vote runs            (default: $JUDGE_RUNS)
  --skip-add              reuse existing corpus, skip ingest
  --output-root <dir>     parent dir for results
                          (default: benchmark_results/run_<ts>)
  --concurrency <N>       run up to N convs in parallel (default: 1 = serial)
                          per-conv stdout/stderr is redirected to
                          \$OUTPUT_ROOT/conv<i>.log so streams don't interleave
  -h | --help             show this help
  --                      everything after is forwarded to test_locomo.py

Any positional or unknown arg goes through to test_locomo.py untouched.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conv-indices) CONV_INDICES="$2"; shift 2 ;;
    --methods)      METHODS="$2"; shift 2 ;;
    --base-url)     BASE_URL="$2"; shift 2 ;;
    --data-path)    DATA_PATH="$2"; shift 2 ;;
    --top-k)        TOP_K="$2"; shift 2 ;;
    --eval-owner)   EVAL_OWNER="$2"; shift 2 ;;
    --judge-runs)   JUDGE_RUNS="$2"; shift 2 ;;
    --skip-add)     SKIP_ADD="true"; shift ;;
    --output-root)  OUTPUT_ROOT="$2"; shift 2 ;;
    --concurrency)  CONCURRENCY="$2"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    --)             shift; EXTRA_ARGS+=("$@"); break ;;
    *)              EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# ── Expand conv-indices spec ──────────────────────────────────────────
expand_indices() {
  local spec="$1"
  if [[ "$spec" == "all" ]]; then
    echo "0 1 2 3 4 5 6 7 8 9"
    return
  fi
  if [[ "$spec" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}"
    return
  fi
  echo "$spec" | tr ',' ' '
}

INDICES=$(expand_indices "$CONV_INDICES")
TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="${OUTPUT_ROOT:-benchmark_results/run_${TS}}"
mkdir -p "$OUTPUT_ROOT"

# ── Plan banner ───────────────────────────────────────────────────────
echo "═════════════════════════════════════════════════════════════════"
echo "  LoCoMo batch run"
echo "═════════════════════════════════════════════════════════════════"
printf "  base_url        : %s\n" "$BASE_URL"
printf "  conv_indices    : %s\n" "$(echo "$INDICES" | tr '\n' ' ')"
printf "  methods         : %s\n" "$METHODS"
printf "  top_k           : %s\n" "$TOP_K"
printf "  eval_owner      : %s\n" "$EVAL_OWNER"
printf "  judge_runs      : %s\n" "$JUDGE_RUNS"
printf "  skip_add        : %s\n" "$SKIP_ADD"
printf "  concurrency     : %s\n" "$CONCURRENCY"
printf "  output_root     : %s\n" "$OUTPUT_ROOT"
[[ ${#EXTRA_ARGS[@]} -gt 0 ]] && printf "  forwarded args  : %s\n" "${EXTRA_ARGS[*]}"
echo

# ── Build per-conv command and launch ────────────────────────────────
#
# bash 3.2 (macOS default) lacks namerefs (`local -n`) and `wait -n`, so
# build_cmd populates a global array CMD and the parallel scheduler
# uses a poll-loop with `kill -0` instead of `wait -n`.
build_cmd() {
  local _ci="$1"
  CMD=(
    PYTHONPATH=src
    python tests/test_locomo.py
    --base-url        "$BASE_URL"
    --data-path       "$DATA_PATH"
    --conv-index      "$_ci"
    --methods         "$METHODS"
    --top-k           "$TOP_K"
    --eval-owner      "$EVAL_OWNER"
    --judge-runs      "$JUDGE_RUNS"
    --output          "$OUTPUT_ROOT/conv${_ci}.json"
    --checkpoint-dir  "$OUTPUT_ROOT/conv${_ci}_checkpoints"
    --corpus-path     "$CORPUS_PATH"
    --post-flush-wait "$POST_FLUSH_WAIT"
    --quiet
  )
  [[ "$SKIP_ADD" == "true" ]] && CMD+=( --skip-add )
  [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && CMD+=( "${EXTRA_ARGS[@]}" )
  # Final no-op: the trailing [[ ]] above can be false (e.g. no extra
  # args), which would make the function's exit status non-zero and
  # trip `set -e` in the caller. Explicit success keeps the contract.
  return 0
}

FAILED=()

if [[ "$CONCURRENCY" -le 1 ]]; then
  # ── Serial path (legacy behaviour) ──────────────────────────────────
  for CI in $INDICES; do
    echo "═════════════════════════════════════════════════════════════════"
    echo "  conv $CI  →  $OUTPUT_ROOT/conv${CI}.json"
    echo "═════════════════════════════════════════════════════════════════"
    build_cmd "$CI"
    set +e
    env "${CMD[@]}"
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
      FAILED+=("conv${CI}")
    fi
  done
else
  # ── Parallel path: job pool of $CONCURRENCY workers ─────────────────
  #
  # Each conv runs in its own python process, streaming to a per-conv
  # log file (conv<i>.log) so interleaved stdout doesn't turn into
  # confetti. Status is collected via `wait $pid`; one conv's failure
  # does not abort the rest.
  echo "─────────────────────────────────────────────────────────────────"
  echo "  Parallel mode: up to $CONCURRENCY convs concurrent"
  echo "  Per-conv logs: $OUTPUT_ROOT/conv<i>.log"
  echo "─────────────────────────────────────────────────────────────────"

  # Parallel arrays (no associative arrays in bash 3.2).
  RUN_PIDS=()
  RUN_CIS=()

  # Wait for *any* worker to exit, reap it, prune the slot, record
  # failures. Polls because `wait -n` is bash 4.3+.
  reap_one() {
    while true; do
      local idx
      for idx in "${!RUN_PIDS[@]}"; do
        local pid="${RUN_PIDS[$idx]}"
        if ! kill -0 "$pid" 2>/dev/null; then
          set +e
          wait "$pid"
          local rc=$?
          set -e
          local ci="${RUN_CIS[$idx]}"
          if [[ $rc -eq 0 ]]; then
            echo "  ✓ conv${ci} done (pid $pid)"
          else
            echo "  ✗ conv${ci} failed (pid $pid, status $rc) — see $OUTPUT_ROOT/conv${ci}.log"
            FAILED+=("conv${ci}")
          fi
          unset 'RUN_PIDS[idx]'
          unset 'RUN_CIS[idx]'
          # Re-pack arrays so ${#RUN_PIDS[@]} stays accurate.
          RUN_PIDS=("${RUN_PIDS[@]}")
          RUN_CIS=("${RUN_CIS[@]}")
          return 0
        fi
      done
      sleep 2
    done
  }

  for CI in $INDICES; do
    build_cmd "$CI"
    LOG="$OUTPUT_ROOT/conv${CI}.log"
    echo "  → launching conv${CI}  (log: $LOG)"
    env "${CMD[@]}" > "$LOG" 2>&1 &
    pid=$!
    RUN_PIDS+=("$pid")
    RUN_CIS+=("$CI")

    if [[ ${#RUN_PIDS[@]} -ge $CONCURRENCY ]]; then
      reap_one
    fi
  done

  # Drain the remaining workers.
  while [[ ${#RUN_PIDS[@]} -gt 0 ]]; do
    reap_one
  done
fi

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo
  echo "⚠ ${#FAILED[@]} conv(s) failed: ${FAILED[*]}"
fi

# ── Aggregate summary ─────────────────────────────────────────────────
echo
echo "═════════════════════════════════════════════════════════════════"
echo "  Aggregate accuracy"
echo "═════════════════════════════════════════════════════════════════"
python - <<EOF
import json
from pathlib import Path

root = Path("$OUTPUT_ROOT")
files = sorted(root.glob("conv*.json"))
if not files:
    print("  (no result files found)")
    raise SystemExit

# header
methods_seen = []
for p in files:
    d = json.load(open(p))
    for m in d["methods"]:
        if m not in methods_seen:
            methods_seen.append(m)

w = max(20, max(len(p.stem) + 4 for p in files))
header = f"{'conversation':<{w}} " + "  ".join(f"{m:>10}" for m in methods_seen)
print(header)
print("─" * len(header))

for p in files:
    d = json.load(open(p))
    label = p.stem
    cells = []
    for m in methods_seen:
        mr = d["methods"].get(m)
        if mr is None:
            cells.append(f"{'—':>10}")
        else:
            raw = mr["summary"]["accuracy"]
            acc = float(str(raw).rstrip("%")) if isinstance(raw, str) else float(raw) * 100
            cells.append(f"{acc:>9.1f}%")
    print(f"{label:<{w}} " + "  ".join(cells))

print()
print(f"  detailed JSONs: {root}/conv*.json")
print(f"  phase checkpoints: {root}/conv*_checkpoints/")
EOF
