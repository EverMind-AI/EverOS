#!/bin/bash
# HyperMem Evaluation Pipeline
# 修改下方参数后直接运行: bash scripts/run_eval.sh

# ==================== Stage Control ====================
STAGES="2 3 4 5 6"                                         # 默认要运行的阶段

# ==================== Experiment Config ====================
export HYPERMEM_EXPERIMENT_NAME="${HYPERMEM_EXPERIMENT_NAME:-base}"
export HYPERMEM_USE_RERANKER="${HYPERMEM_USE_RERANKER:-false}"
export HYPERMEM_INITIAL_CANDIDATES="${HYPERMEM_INITIAL_CANDIDATES:-100}"
export HYPERMEM_TOPIC_TOP_K="${HYPERMEM_TOPIC_TOP_K:-10}"
export HYPERMEM_EPISODE_TOP_K="${HYPERMEM_EPISODE_TOP_K:-10}"
export HYPERMEM_FACT_TOP_K="${HYPERMEM_FACT_TOP_K:-30}"

# ==================== Environment ====================
CONDA_ENV="hypermem"

# ===========================================================

eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

cd "$(dirname "$0")/.."

# If no arguments passed, use STAGES; otherwise use passed arguments
if [ $# -eq 0 ]; then
    EVAL_ARGS="--stages $STAGES"
else
    EVAL_ARGS="$@"
fi

echo "========================================"
echo "  HyperMem Evaluation Pipeline"
echo "========================================"
echo "  Conda env:   $CONDA_ENV"
echo "  Experiment:  $HYPERMEM_EXPERIMENT_NAME"
echo "  Reranker:    $HYPERMEM_USE_RERANKER"
echo "  Config:      ${HYPERMEM_INITIAL_CANDIDATES}-${HYPERMEM_TOPIC_TOP_K}-${HYPERMEM_EPISODE_TOP_K}-${HYPERMEM_FACT_TOP_K}"
echo "  Eval args:   $EVAL_ARGS"
echo "========================================"

python hypermem/main/eval.py $EVAL_ARGS
