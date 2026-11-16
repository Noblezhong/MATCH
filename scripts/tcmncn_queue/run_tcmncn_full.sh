#!/usr/bin/env bash
# Full 1418-episode run of the Table-3 cell "w/ TCM + NCN" on 5090-LSQ.
source "${TCMNCN_ENV_FILE:-/home/smc/match5090/env.sh}"
ROOT_DIR="${MATCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export EVAL_SAVE_PATH="${EVAL_SAVE_PATH:-$ROOT_DIR/eval_continuous_tcmncn_seen1418_20260918}"
export DDP_MASTER_PORT=80009   # distinct from the w/NCN run's 80008
cd "$ROOT_DIR"
exec bash scripts/continue_tcm_ncn_eval.sh 2>&1 | tee -a /tmp/tcmncn_full_run.log
