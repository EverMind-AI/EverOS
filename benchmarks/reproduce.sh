#!/usr/bin/env bash
# Full reproduction of a benchmark at its best-known configuration.
#
# Everything that decides the number -- backbone, decider, answer and judge models,
# top_k, retrieval knobs, concurrency -- lives in benchmarks/configs/<dataset>.toml, and
# this script passes no model overrides. Servers, ports and store paths are derived from
# the run name.
set -euo pipefail

# ---------------------------------------------------------------- edit here
# Exported, not just assigned: the banner below runs in a child process and reads these
# from the environment. Plain assignment left it reporting `locomo` while the run used
# whatever was edited here -- a banner that disagrees with the run is worse than none.
export DATASET="${DATASET:-locomo}"   # locomo | longmemeval | subtlememory | evermembench
export CONV="${CONV:-all}"            # full set; a slice is not a reproduction
export STAGES="${STAGES:-add search answer judge}"
# --------------------------------------------------------------------------

# Result directory name: the bare dataset for a full run, <dataset>_conv<range> for a
# slice, so the name alone says whether the number is comparable to a published figure.
# Spaces in a conv list are squashed: `CONV="0 1"` produced a directory literally called
# `locomo_conv0 1`, and every path derived from it carried the space.
_conv_tag="${CONV// /-}"
export RUN="${RUN:-$([ "$CONV" = "all" ] && echo "$DATASET" || echo "${DATASET}_conv${_conv_tag}")}"

# Derived from this script's own location, so a clone works wherever it lands.
# One level, not two: this file sits in benchmarks/, and `../..` was right only while it
# lived in benchmarks/scripts/. Moving it up without changing this made it cd to the
# repository's PARENT, where it then reported benchmarks/.env missing -- from a tree that
# had one.
EVEROS=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$EVEROS"

# A missing .env used to abort here with a bare "No such file or directory" from the
# shell, which is a fresh clone's first command and says nothing about what to do.
if [ ! -f benchmarks/.env ]; then
  echo "benchmarks/.env not found." >&2
  echo "  cp benchmarks/.env.example benchmarks/.env" >&2
  echo "  then fill in the answer/judge keys and the BENCH_DATA_* path for $DATASET." >&2
  exit 1
fi
set -a; . benchmarks/.env; set +a

echo "dataset  $DATASET"
echo "conv     $CONV"
echo "run      $RUN"
.venv/bin/python - <<'PYEOF'
import os
import sys

sys.path.insert(0, "benchmarks")
from config import BenchmarkConfig, unresolved

c = BenchmarkConfig.from_toml(os.environ.get("DATASET", "locomo"))
# `unresolved`, not truthiness: the shipped configs name the decider as
# ${BENCH_DECIDER_MODEL} so nothing in them is specific to one deployment, and an unset
# variable arrives here as that literal. Printing it verbatim reported a model nobody is
# serving; the run itself treats it as "no separate decider" and uses [llm].
_dec = c.decider_model
print(f"backbone {c.backbone_model}")
print(f"decider  {'(= backbone)' if not _dec or unresolved(_dec) else _dec}")
print(f"answer   {c.answer_model}")
print(f"judge    {c.judge_model} x{c.judge_runs}")
print(f"method   {c.methods} top_k={c.top_k} servers={c.servers}")
for field, value in (("data_path", c.data_path), ("results_root", c.results_root)):
    if unresolved(value):
        print(f"WARNING  {field} is unresolved: {value}")
PYEOF

# $CONV and $STAGES unquoted on purpose: both are nargs="+" lists. Quoting sent the whole
# list as ONE argv entry, and argparse then handed "0 1" to int().
#
# "$@" is forwarded so anything run.py accepts can be passed through. Without it this
# script's own failure was a dead end: a STAGES=search run prints "--everos-root is
# required", and there was no way to supply it here.
exec .venv/bin/python benchmarks/run.py "$DATASET" \
  --run-name "$RUN" --conv $CONV --stages $STAGES "$@"
