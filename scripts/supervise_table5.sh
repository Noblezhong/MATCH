#!/usr/bin/env bash
# Serial supervisor for the 4 Table-5 rows (5090).
# Per row: restart the AirSim server before each attempt, run the row
# (resume-safe: completed episodes are skipped by the evaluator), stop when
# the save dir holds >=1420 entries (1418 episode dirs + profile_logs +
# manifest) or when two consecutive attempts add zero episodes (abort row).
# Logs: $ROOT_DIR/table5_logs/<row>.log
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${TABLE5_LOG_DIR:-$ROOT_DIR/table5_logs}"
STAMP="${TABLE5_STAMP:-$(date +%m%d)}"
mkdir -p "$LOG_DIR"

log() { echo "$(date '+%F %T') $*"; }

restart_airsim() {
  tmux kill-session -t airsim5090 2>/dev/null || true
  sleep 2
  tmux new-session -d -s airsim5090 "bash $ROOT_DIR/scripts/airsim_server_5090.sh 2>&1 | tee -a $LOG_DIR/airsim_server.log"
  log "AirSim ServerTool restarting (tmux airsim5090), waiting 45 s for readiness"
  sleep 45
}

count_entries() { ls "$1" 2>/dev/null | wc -l; }

run_row() {
  local row="$1" save="$2" trace="$3" port="$4"
  local attempt=0 stalled=0 before after
  while true; do
    before=$(count_entries "$save")
    if (( before >= 1420 )); then
      log "[$row] COMPLETE ($before entries)"
      return 0
    fi
    attempt=$((attempt + 1))
    log "[$row] attempt $attempt (have $before entries)"
    restart_airsim
    TABLE5_ROW="$row" TABLE5_TRACE="$trace" TABLE5_SAVE="$save" TABLE5_DDP_PORT="$port" \
      bash "$ROOT_DIR/scripts/run_table5_row.sh" >> "$LOG_DIR/${row}.log" 2>&1
    after=$(count_entries "$save")
    log "[$row] attempt $attempt exited: $before -> $after entries"
    if (( after <= before )); then
      stalled=$((stalled + 1))
      if (( stalled >= 2 )); then
        log "[$row] ABORT after $attempt attempts: no progress in $stalled consecutive attempts"
        return 1
      fi
    else
      stalled=0
    fi
  done
}

D="$ROOT_DIR"
run_row "eo_lc"    "$D/eval_match_edge_only_pilot11_0909_lc1418_${STAMP}_fast_x10"     "$D/bandwidth/low_constrained_trace.csv" 80022
run_row "eo_out"   "$D/eval_match_edge_only_pilot11_0909_outage1418_${STAMP}_fast_x10" "$D/bandwidth/real_outage_trace.csv"     80022
run_row "match_lc" "$D/eval_match_full_pilot11_0909_lc1418_${STAMP}_fast_x10"          "$D/bandwidth/low_constrained_trace.csv" 80023
run_row "match_out" "$D/eval_match_full_pilot11_0909_outage1418_${STAMP}_fast_x10"     "$D/bandwidth/real_outage_trace.csv"     80023
log "ALL TABLE5 ROWS DONE"
