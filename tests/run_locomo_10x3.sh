#!/usr/bin/env bash
# Run the LoCoMo benchmark across all 10 conversations × 3 retrieval
# methods (keyword, vector, hybrid).
#
# Wraps tests/run_locomo_batch.sh with the defaults that match the
# everos post-fix benchmark protocol:
#   - all 10 LoCoMo conversations (conv 0..9)
#   - keyword + vector + hybrid (agentic is skipped — costs 2-3× more
#     LLM tokens and the rerank loop hasn't been benchmarked yet)
#   - speaker_a partition (the LoCoMo "Plan C" single-owner eval)
#   - judge runs = 1 (single-pass LLM judge, no majority vote)
#   - top-K 10
#
# Two ingest modes:
#
#   --skip-add (default)        reuse the corpus that already lives at
#                               ~/.everos-report-corpus. Skips the
#                               ~5 min/conv ingest phase × 10 = ~50 min
#                               saved. Note: the existing corpus may
#                               still carry artefacts from the OLD code
#                               (conv-5 missing episode rows,
#                               MRAG score=0.0 facts). For a strictly
#                               clean benchmark of the *fixed* code,
#                               use --fresh-corpus instead.
#
#   --fresh-corpus              wipe ~/.everos-report-corpus, restart
#                               the server, and re-ingest every conv
#                               with the current bug-fixed cascade.
#                               Adds ~50 min to the run.
#
# Server must already be running on :8000 with the current code loaded
# (i.e. the OR + optimize fixes). Health check confirmed before launch.
#
# Output structure:
#
#   benchmark_results/run_<ts>_10x3/
#   ├── conv0.json  ...  conv9.json     ← per-conv final results
#   ├── conv0_checkpoints/  ...         ← phase-level snapshots
#   └── SUMMARY.md                       ← cross-conv accuracy table

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd)"

# ── Defaults ──────────────────────────────────────────────────────────
BASE_URL="${BASE_URL:-http://localhost:8000}"
DATA_PATH="${DATA_PATH:-data/locomo10.json}"
MEMORY_ROOT="${EVEROS_MEMORY__ROOT:-$HOME/.everos-report-corpus}"
MODE="skip-add"        # default; toggle via --fresh-corpus
TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_ROOT="$REPO_ROOT/benchmark_results/run_${TS}_10x3"

# ── Parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-add)        MODE="skip-add"; shift ;;
    --fresh-corpus)    MODE="fresh"; shift ;;
    --base-url)        BASE_URL="$2"; shift 2 ;;
    --memory-root)     MEMORY_ROOT="$2"; shift 2 ;;
    --output-root)     OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help)
      grep -E "^# " "$0" | sed 's/^# //;s/^#//'
      exit 0
      ;;
    *)
      echo "unknown arg: $1" >&2
      exit 1
      ;;
  esac
done

# ── Preflight ─────────────────────────────────────────────────────────
echo "═════════════════════════════════════════════════════════════════"
echo "  LoCoMo 10 × 3 benchmark"
echo "═════════════════════════════════════════════════════════════════"
echo "  mode:        $MODE"
echo "  base_url:    $BASE_URL"
echo "  memory_root: $MEMORY_ROOT"
echo "  output:      $OUTPUT_ROOT"
echo

# 1. Server up?
if ! curl -fsS -o /dev/null "$BASE_URL/health" 2>/dev/null; then
  echo "❌ server at $BASE_URL is not responding"
  echo "   start with: EVEROS_MEMORY__ROOT=$MEMORY_ROOT PYTHONPATH=src \\"
  echo "     python -m everos.entrypoints.cli.main server start --port 8000"
  exit 1
fi
echo "✓ server healthy"

# 2. LLM env (test_locomo.py reads bare LLM_* — bridge from EVEROS_LLM__*)
if [[ -z "${LLM_API_KEY:-}" ]] || [[ -z "${LLM_BASE_URL:-}" ]] || [[ -z "${LLM_MODEL:-}" ]]; then
  if [[ -f "$REPO_ROOT/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source <(grep -E "^EVEROS_LLM__" "$REPO_ROOT/.env" | sed 's/EVEROS_LLM__/LLM_/')
    set +a
  fi
fi
if [[ -z "${LLM_API_KEY:-}" ]]; then
  echo "❌ LLM_API_KEY not set (and .env has no EVEROS_LLM__API_KEY to bridge from)"
  exit 1
fi
echo "✓ LLM credentials: model=$LLM_MODEL @ $LLM_BASE_URL"
echo

# 3. Fresh corpus mode → wipe + restart server
if [[ "$MODE" == "fresh" ]]; then
  echo "═════════════════════════════════════════════════════════════════"
  echo "  --fresh-corpus: wiping $MEMORY_ROOT and restarting server"
  echo "═════════════════════════════════════════════════════════════════"

  # Find and kill existing server (best-effort)
  pids="$(pgrep -f "everos.entrypoints.cli.main server" || true)"
  if [[ -n "$pids" ]]; then
    echo "  stopping server pid(s): $pids"
    # shellcheck disable=SC2086
    kill $pids
    sleep 3
  fi

  rm -rf "$MEMORY_ROOT"
  mkdir -p "$MEMORY_ROOT"

  # Restart in background; the server picks up the empty memory root.
  echo "  starting fresh server..."
  (
    cd "$REPO_ROOT"
    EVEROS_MEMORY__ROOT="$MEMORY_ROOT" \
    PYTHONPATH=src \
    nohup python -m everos.entrypoints.cli.main server start --port 8000 \
      > /tmp/everos-server-${TS}.log 2>&1 &
    echo "  server pid=$!"
  )

  # Wait for lifespan ready
  for i in $(seq 1 60); do
    if curl -fsS -o /dev/null "$BASE_URL/health" 2>/dev/null; then
      echo "  server ready after ${i}s"
      break
    fi
    sleep 1
  done
  if ! curl -fsS -o /dev/null "$BASE_URL/health" 2>/dev/null; then
    echo "❌ server failed to come up; see /tmp/everos-server-${TS}.log"
    exit 1
  fi
  echo
fi

# ── Build the batch invocation ────────────────────────────────────────
BATCH_ARGS=(
  --conv-indices 0-9
  --methods keyword,vector,hybrid
  --base-url "$BASE_URL"
  --top-k 10
  --eval-owner speaker_a
  --judge-runs 1
  --output-root "$OUTPUT_ROOT"
)
[[ "$MODE" == "skip-add" ]] && BATCH_ARGS+=( --skip-add )

echo "═════════════════════════════════════════════════════════════════"
echo "  Launching: tests/run_locomo_batch.sh ${BATCH_ARGS[*]}"
echo "═════════════════════════════════════════════════════════════════"
echo

cd "$REPO_ROOT"
bash tests/run_locomo_batch.sh "${BATCH_ARGS[@]}"

# ── Summary markdown ──────────────────────────────────────────────────
echo
echo "═════════════════════════════════════════════════════════════════"
echo "  Rendering SUMMARY.md"
echo "═════════════════════════════════════════════════════════════════"

python - <<PYEOF
import json
from pathlib import Path

root = Path("$OUTPUT_ROOT")
out_md = root / "SUMMARY.md"
files = sorted(root.glob("conv*.json"))
if not files:
    print(f"no result files under {root}")
    raise SystemExit

methods_seen: list[str] = []
for p in files:
    d = json.load(open(p))
    for m in d["methods"]:
        if m not in methods_seen:
            methods_seen.append(m)

cat_names = {"1": "single-hop", "2": "multi-hop", "3": "open-domain", "4": "temporal"}

lines: list[str] = []
lines.append(f"# LoCoMo 10×3 — run_${TS}\n")
lines.append(
    f"- mode: `{'$MODE'}`\n"
    f"- base_url: \`$BASE_URL\`\n"
    f"- memory_root: \`$MEMORY_ROOT\`\n"
    f"- methods: \`{', '.join(methods_seen)}\`\n"
)

# Per-conv table
lines.append("\n## Per-conv accuracy\n\n")
lines.append("| conv | " + " | ".join(f"**{m}**" for m in methods_seen) + " |\n")
lines.append("|---|" + "|".join(["---"] * len(methods_seen)) + "|\n")

agg_correct = {m: 0 for m in methods_seen}
agg_total = {m: 0 for m in methods_seen}
cat_correct: dict[str, dict[str, int]] = {m: {} for m in methods_seen}
cat_total: dict[str, dict[str, int]] = {m: {} for m in methods_seen}

for p in files:
    d = json.load(open(p))
    cells = []
    for m in methods_seen:
        mr = d["methods"].get(m)
        if mr is None:
            cells.append("—")
            continue
        s = mr["summary"]
        cells.append(f"{s['accuracy']*100:.1f}%")
        agg_correct[m] += s["correct"]
        agg_total[m] += s["total"]
        for cat, st in s["category_stats"].items():
            cat_correct[m][cat] = cat_correct[m].get(cat, 0) + st["correct"]
            cat_total[m][cat] = cat_total[m].get(cat, 0) + st["total"]
    lines.append(f"| {p.stem} | " + " | ".join(cells) + " |\n")

# Overall
overall = []
for m in methods_seen:
    if agg_total[m]:
        overall.append(f"**{agg_correct[m]/agg_total[m]*100:.1f}%**")
    else:
        overall.append("—")
lines.append(f"| **OVERALL** | " + " | ".join(overall) + " |\n")

# Per-category
lines.append("\n## Per-category accuracy (across all 10 convs)\n\n")
lines.append("| cat | kind | " + " | ".join(f"**{m}**" for m in methods_seen) + " |\n")
lines.append("|---|---|" + "|".join(["---"] * len(methods_seen)) + "|\n")
for cat in ["1", "2", "3", "4"]:
    cells = []
    for m in methods_seen:
        tot = cat_total[m].get(cat, 0)
        if tot:
            cells.append(f"{cat_correct[m][cat] / tot * 100:.1f}%")
        else:
            cells.append("—")
    lines.append(f"| {cat} | {cat_names[cat]} | " + " | ".join(cells) + " |\n")

out_md.write_text("".join(lines), encoding="utf-8")
print(f"  → {out_md}")
print()
print("".join(lines))
PYEOF

echo
echo "Done."
