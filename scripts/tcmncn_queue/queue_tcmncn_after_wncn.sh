#!/usr/bin/env bash
# Queue the Table-3 cell "w/ TCM + NCN" behind the running "w/ NCN" run.
#
#   1. wait until w/NCN has 1418 episode dirs AND its process is gone
#   2. launch the TCM+NCN run in tmux (supervised + resume-aware)
#   3. validate the first episodes; kill the session on a hard failure
#
# Nothing here touches the w/NCN run or its directory.
set -uo pipefail

REPO="${MATCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
WNCN_DIR="$REPO/eval_continuous_ncn_seen1418_20260917_fast_x10"
TCM_DIR="$REPO/eval_continuous_tcmncn_seen1418_20260918_fast_x10"
RUN_LOG=/tmp/tcmncn_full_run.log
TARGET=1418
SESSION=tcmncn1418
STATUS="$REPO/tcmncn_queue_status.log"
POLL=120
STALL_LIMIT=5

mkdir -p "$REPO"
count_dirs() { find "$1" -maxdepth 1 -mindepth 1 -type d ! -name profile_logs 2>/dev/null | wc -l; }
ncn_alive()  { ps -eo args --no-headers | grep -q "[c]ontinue_ncn_eval.py"; }
log()        { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

# ── 1. wait for w/NCN ────────────────────────────────────────────────────
log "queue armed — waiting for w/NCN to reach $TARGET episodes"
stall=0; prev=-1
while true; do
  n=$(count_dirs "$WNCN_DIR")
  if ncn_alive; then
    log "w/NCN running: $n/$TARGET"
    stall=0
  elif [ "$n" -ge "$TARGET" ]; then
    log "w/NCN COMPLETE: $n/$TARGET, process gone"
    break
  else
    # process gone but the episode count is short — do NOT launch on top of a
    # half-finished run; report and stop.
    if [ "$n" -eq "$prev" ]; then stall=$((stall+1)); else stall=0; fi
    log "w/NCN NOT running but only $n/$TARGET (stall $stall/$STALL_LIMIT)"
    if [ "$stall" -ge "$STALL_LIMIT" ]; then
      log "ABORT: w/NCN stopped at $n/$TARGET without progressing — resume it before queueing this cell"
      exit 1
    fi
  fi
  prev=$n
  sleep "$POLL"
done

# ── 2. launch ────────────────────────────────────────────────────────────
log "launching TCM+NCN in tmux session $SESSION"
tmux kill-session -t "$SESSION" 2>/dev/null
tmux new-session -d -s "$SESSION" "bash $REPO/scripts/tcmncn_queue/supervise_tcmncn.sh"
sleep 30

# ── 3. validate the first episodes ───────────────────────────────────────
fail() {
  log "SMOKE FAIL: $*"
  tmux kill-session -t "$SESSION" 2>/dev/null
  log "session $SESSION killed — the full cell was NOT left running"
  exit 1
}

deadline=$(( $(date +%s) + 2700 ))   # 45 min to get >=3 episodes
jsonl=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  jsonl=$(ls -t "$TCM_DIR"/profile_logs/pro_con_w5_*.jsonl 2>/dev/null | head -1)
  if [ -n "$jsonl" ]; then
    ep=$(grep -c '"record_type": "episode_end"' "$jsonl" 2>/dev/null || echo 0)
    if [ "$ep" -ge 3 ]; then break; fi
  fi
  sleep 30
done

[ -n "$jsonl" ] || fail "no profile JSONL produced within 45 min"
[ -f "$jsonl" ] || fail "profile JSONL path vanished: $jsonl"

ep=$(grep -c '"record_type": "episode_end"' "$jsonl" || echo 0)
miss=$(grep -c 'missing_coarse_frame' "$jsonl" || echo 0)
corr=$(grep -c '"trajectory_mode": "corrected"' "$jsonl" || echo 0)
locks=$(grep -c '"target_lock_active": true' "$jsonl" || echo 0)
tb=$(grep -c 'Traceback' "$RUN_LOG" 2>/dev/null || echo 0)
ncnact=$(grep -c '"control_source": "ncn"' "$jsonl" || echo 0)

log "smoke stats: episodes=$ep corrected_applies=$corr lock_records=$locks ncn_actions=$ncnact"
log "smoke stats: missing_coarse_frame=$miss tracebacks=$tb"

[ "$tb" -eq 0 ]   || fail "$tb Traceback(s) in $RUN_LOG"
[ "$miss" -eq 0 ] || fail "$miss missing_coarse_frame fallback(s) — the coarse_local/coarse_goal_world fields are not being populated, TCM would be inert"
[ "$corr" -gt 0 ] || fail "TCM never fired in $ep episodes (0 corrected applies) — trajcorr_mode is not reaching the loop"
[ "$ep" -ge 3 ]   || fail "only $ep episode(s) completed"

log "SMOKE PASS — TCM and NCN both live. Full cell continues in tmux $SESSION (resumes automatically, target $TARGET)."
