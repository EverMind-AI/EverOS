#!/bin/bash
# One search throughput sweep: a fresh EverOS server on a copy of the store, the
# stub model providers in front of it, a 30 s warm-up at 2 requests/s, then
# load.py at rising rates until two steps fall behind. See
# benchmarks/throughput.md.
#
# usage: sweep.sh <everos> <store> <out_dir> <hybrid|keyword>
#   everos   the `everos` executable of the version under test (its venv's python
#            also runs the stub and the load unless PY is set)
#   store    an EverOS root holding app perf / project real100k (build_store.py);
#            copied to <out_dir>/store first, so the original stays untouched
# env:
#   QUERIES      (user_id, query) lines (default: <store>/../queries.jsonl)
#   RATES        requests per second to step through
#                (default hybrid: 4 5 6 8 10 12 14 16 20 24 28 32,
#                         keyword: 40 60 80 100 120 140 160 180 200)
#   STEP_SECONDS per step (default hybrid 60, keyword 30)
#   SERVER_WRAP  command prefix for the server, e.g. core pinning and limits:
#                "systemd-run --user --scope --quiet -p MemoryMax=8G -p CPUQuota=400% taskset -c 0-3"
#   CLIENT_WRAP  command prefix for the stub and the load, e.g. "taskset -c 4-7"
#   SERVER_ENV   extra environment for the server, e.g. "LD_PRELOAD="
#   WAIT_INDEX   vector index type to wait for before measuring (IvfRq: a store
#                written before 1.8 is rebuilt to IVF_RQ by the cascade's
#                maintenance beat after start-up, which takes minutes)
set -u
EVEROS=$1; STORE=$(cd "$2" && pwd); OUT=$3; METHOD=$4
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$(dirname "$EVEROS")/python}
QUERIES=${QUERIES:-$STORE/../queries.jsonl}
case $METHOD in
  hybrid) RATES=${RATES:-"4 5 6 8 10 12 14 16 20 24 28 32"}; STEP=${STEP_SECONDS:-60} ;;
  keyword) RATES=${RATES:-"40 60 80 100 120 140 160 180 200"}; STEP=${STEP_SECONDS:-30} ;;
  *) echo "method must be hybrid or keyword"; exit 2 ;;
esac
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$OUT/sweep.log"; }
free_port() { "$PY" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
STUB=""; SRV=""
cleanup() {
  for p in $SRV $STUB; do kill -TERM "$p" 2>/dev/null; done
  sleep 5
  for p in $SRV $STUB; do kill -KILL "$p" 2>/dev/null; done
}
trap cleanup EXIT
trap 'exit 143' INT TERM HUP

if [ ! -d "$OUT/store" ]; then
  log "copying $STORE"
  if [ "$(uname)" = Darwin ]; then cp -c -R "$STORE" "$OUT/store"; else cp -R --reflink=auto "$STORE" "$OUT/store"; fi
fi

MP=$(free_port)
${CLIENT_WRAP:-} "$PY" "$HERE/stub_models.py" "$MP" > "$OUT/stub.log" 2>&1 &
STUB=$!
for _ in $(seq 1 100); do grep -q 'stub models on' "$OUT/stub.log" && break; sleep 0.2; done
SP=$(free_port)
M=http://127.0.0.1:$MP
# shellcheck disable=SC2086
env -u VIRTUAL_ENV EVEROS_ROOT="$OUT/store" EVEROS_API__HOST=127.0.0.1 EVEROS_API__PORT="$SP" \
  EVEROS_LLM__BASE_URL=$M/llm EVEROS_LLM__API_KEY=stub EVEROS_LLM__MODEL=stub \
  EVEROS_EMBEDDING__BASE_URL=$M/emb EVEROS_EMBEDDING__API_KEY=stub \
  EVEROS_EMBEDDING__MODEL=Qwen/Qwen3-Embedding-4B EVEROS_EMBEDDING__DIM=1024 \
  EVEROS_RERANK__PROVIDER=vllm EVEROS_RERANK__BASE_URL=$M/rerank \
  EVEROS_RERANK__API_KEY=stub EVEROS_RERANK__MODEL=stub ${SERVER_ENV:-} \
  ${SERVER_WRAP:-} "$EVEROS" server start > "$OUT/server.log" 2>&1 &
SRV=$!
log "everos $("$PY" -c 'import importlib.metadata as m; print(m.version("everos"))'), server pid $SRV port $SP, $METHOD"

for _ in $(seq 1 150); do curl -sf -m 2 "http://127.0.0.1:$SP/health" > /dev/null && break; sleep 2; done
curl -sf -m 5 "http://127.0.0.1:$SP/health" > /dev/null || { log "server never healthy"; tail -20 "$OUT/server.log"; exit 1; }
# A store the server has not indexed yet is reconciled first; measure after that.
calm=0
for _ in $(seq 1 360); do
  pending=$(curl -sf -m 5 "http://127.0.0.1:$SP/health" | "$PY" -c 'import json, sys; print(json.load(sys.stdin).get("cascade", {}).get("pending", -1))')
  if [ "$pending" = 0 ]; then calm=$((calm + 1)); else calm=0; fi
  [ $calm -ge 2 ] && break
  sleep 5
done
if [ -n "${WAIT_INDEX:-}" ]; then
  log "waiting for $WAIT_INDEX vector indexes"
  "$PY" - "$OUT/store/.index/lancedb" "$WAIT_INDEX" <<'EOF' || { log "vector indexes never became $WAIT_INDEX"; exit 1; }
import sys, time
import lancedb
path, want = sys.argv[1], sys.argv[2]
for _ in range(120):
    db = lancedb.connect(path)
    kinds = {
        name: ix.index_type
        for name in db.table_names()
        for ix in db.open_table(name).list_indices()
        if ix.index_type.startswith("Ivf")
    }
    print(kinds, flush=True)
    if kinds and set(kinds.values()) == {want}:
        sys.exit(0)
    time.sleep(15)
sys.exit(1)
EOF
fi
sleep 20

PID_ARG=""
[ -d /proc/$SRV ] && PID_ARG="--pid $SRV"  # CPU and memory sampling needs Linux /proc
# The first queries after start-up open index files and fill caches; warm up at a
# low rate first (kept in warmup/, not counted).
mkdir -p "$OUT/warmup"
# shellcheck disable=SC2086
log "warm-up $(${CLIENT_WRAP:-} "$PY" "$HERE/load.py" --base "http://127.0.0.1:$SP" --method "$METHOD" \
  --rate 2 --seconds 30 --queries "$QUERIES" --seed 1 --out "$OUT/warmup/${METHOD}_2")"
sleep 10
behind=0
for r in $RATES; do
  # shellcheck disable=SC2086
  line=$(${CLIENT_WRAP:-} "$PY" "$HERE/load.py" --base "http://127.0.0.1:$SP" --method "$METHOD" \
    --rate "$r" --seconds "$STEP" --queries "$QUERIES" $PID_ARG --out "$OUT/${METHOD}_$r")
  log "$line"
  case $line in *'"keeps_up": false'*) behind=$((behind + 1)) ;; esac
  [ $behind -ge 2 ] && break
  sleep 10
done
log "done"
