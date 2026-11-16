#!/usr/bin/env bash
# Table 5 single-row launcher (5090).
# Env contract (set by supervise_table5.sh):
#   TABLE5_ROW       eo_lc | eo_ro | match_lc | match_ro
#   TABLE5_TRACE     absolute path to the bandwidth trace csv
#   TABLE5_SAVE      absolute path to the eval save dir
#   TABLE5_DDP_PORT  DDP master port for this row
# AHS checkpoint pinned to pilot11 0909-30k (same as Table 3 rows).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROW="${TABLE5_ROW:?Set TABLE5_ROW}"
TRACE="${TABLE5_TRACE:?Set TABLE5_TRACE}"
SAVE="${TABLE5_SAVE:?Set TABLE5_SAVE}"
PORT="${TABLE5_DDP_PORT:?Set TABLE5_DDP_PORT}"

export NCN_CONDA_ENV=llamauav
export CONDA_SH="${CONDA_SH:-/home/noble/miniforge3/etc/profile.d/conda.sh}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GGML_CUDA_DISABLE_GRAPHS=1
export PYTHONFAULTHANDLER=1
export SCHEDULER_MODEL_PATH="$ROOT_DIR/drl_train_pilot11_0909-2109/scheduler_models/ppo_scheduler_20260909-210915-662188.zip"
export COMM_TRACE_CSV_PATH="$TRACE"
export EVAL_SAVE_PATH="$SAVE"
export DDP_MASTER_PORT="$PORT"
export AERODPO_GGUF_DIR="${AERODPO_GGUF_DIR:-/home/noble/mnt/remote_aerodpo_q8}"

echo "[run_table5_row] row=$ROW trace=$TRACE save=$SAVE port=$PORT"
case "$ROW" in
  eo_*)    exec bash "$ROOT_DIR/scripts/match_edge_only_eval.sh" ;;
  match_*) exec bash "$ROOT_DIR/scripts/match_eval.sh" ;;
  *) echo "unknown TABLE5_ROW: $ROW" >&2; exit 2 ;;
esac
