#!/usr/bin/env bash
# On-machine auto-heal for the Table-5 queue (5090).
#
# Why this exists: the UE renderer dies every few hours, but the evaluator does
# NOT exit -- it limps on stale frames while retrying. supervise_table5.sh only
# restarts AirSim when the evaluator process exits, so without this the queue
# silently crawls (~2 episodes/hour) instead of resuming at full speed.
#
# Two detectors. Both require a live evaluator, and the backstop is checked
# first so the two can never disagree about what to do.
#
#   1. IDLE (backstop): no new episode directory for IDLE_MIN minutes. Stays at
#      50 because the measured per-episode wall time is p50 9 min / p95 43.5
#      min, so 50 is just above the legitimate tail. This is the only detector
#      that catches "UE process alive but the renderer is wedged".
#   2. UE-DEAD (fast path): the UE process was seen alive under the CURRENT
#      evaluator and has now been gone continuously for UE_DEAD_S seconds.
#      A scene reopen (reopen_scenes) also kills and relaunches UE, but that
#      takes ~1-3 min, so the 6-minute default does not fire on it. Firing on
#      this signal costs ~5 min instead of ~50.
#
# The "seen alive under the current evaluator" latch matters: at attempt start
# the evaluator spends ~2-4 min loading LLaVA before it launches the scene, so
# a plain "UE is absent" test would false-fire on every restart.
#
# Tunables (env): IDLE_MIN, UE_DEAD_S, POLL, DRYRUN, LOG.
set -uo pipefail

ROOT="${MATCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IDLE_MIN=${IDLE_MIN:-50}
UE_DEAD_S=${UE_DEAD_S:-360}
POLL=${POLL:-120}
DRYRUN=${DRYRUN:-0}
LOG=${LOG:-$ROOT/table5_logs/autoheal.log}

CAND=(
  eval_match_edge_only_pilot11_0909_lc1418_0923_fast_x10
  eval_match_edge_only_pilot11_0909_outage1418_0923_fast_x10
  eval_match_full_pilot11_0909_lc1418_0923_fast_x10
  eval_match_full_pilot11_0909_outage1418_0923_fast_x10
)

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

fire() {
  local reason="$1"
  local gone_s=0
  if [ "$ue_absent_since" -gt 0 ]; then
    gone_s=$(( $(date +%s) - ue_absent_since ))
  fi
  if [ "$DRYRUN" = "1" ]; then
    log "DRYRUN would fire [$reason]: row=$active dir_age=${age}min ue_seen=$ue_seen ue_gone=${gone_s}s"
    return
  fi
  log "STALL[$reason]: row=$active dir_age=${age}min -> killing evaluator for supervisor restart"
  pkill -9 -f "[m]atch_eval.py"
  sleep 90
  pkill -9 -f "[c]loseloop_envs|[c]arla_town_envs"
  log "killed; supervisor should restart AirSim and resume"
  eval_pid=""
  ue_seen=0
  ue_absent_since=0
  sleep 300
}

log "auto-heal armed (IDLE_MIN=${IDLE_MIN}m, UE_DEAD_S=${UE_DEAD_S}s, POLL=${POLL}s, DRYRUN=${DRYRUN})"

eval_pid=""
ue_seen=0
ue_absent_since=0

while true; do
  sleep "$POLL"
  now=$(date +%s)

  best=0
  active=""
  for d in "${CAND[@]}"; do
    [ -d "$ROOT/$d" ] || continue
    m=0
    for p in "$ROOT/$d"/*/; do
      [ -d "$p" ] || continue
      t=$(stat -c %Y "$p" 2>/dev/null) || continue
      [ "$t" -gt "$m" ] && m=$t
    done
    if [ "$m" -gt "$best" ]; then
      best=$m
      active=$d
    fi
  done
  [ -z "$active" ] && continue

  # Bracket forms so this script never matches its own command line.
  # Scope recovery to the Table-5 evaluator whose --eval_save_path matches
  # the selected row. A different MATCH run may share this host and must not
  # be terminated by this table-specific watcher.
  pid=""
  for candidate_pid in $(pgrep -f "[m]atch_eval.py" 2>/dev/null | sort -n); do
    [ -r "/proc/${candidate_pid}/cmdline" ] || continue
    command_line=$(tr '\0' ' ' < "/proc/${candidate_pid}/cmdline")
    if [[ "$command_line" == *"$ROOT/$active"* ]]; then
      pid="$candidate_pid"
      break
    fi
  done
  if [ -z "$pid" ]; then
    eval_pid=""
    ue_seen=0
    ue_absent_since=0
    continue
  fi
  if [ "$pid" != "$eval_pid" ]; then
    log "evaluator pid=$pid (new attempt); UE latch reset"
    eval_pid="$pid"
    ue_seen=0
    ue_absent_since=0
  fi

  if pgrep -f "[c]loseloop_envs|[c]arla_town_envs" >/dev/null 2>&1; then
    if [ "$ue_seen" -eq 0 ]; then
      log "UE up under evaluator pid=$pid"
    fi
    ue_seen=1
    ue_absent_since=0
  elif [ "$ue_seen" -eq 1 ] && [ "$ue_absent_since" -eq 0 ]; then
    ue_absent_since=$now
    log "UE gone while evaluator pid=$pid is alive; watching (fires at ${UE_DEAD_S}s)"
  fi

  age=$(( (now - best) / 60 ))
  if [ "$age" -ge "$IDLE_MIN" ]; then
    fire "IDLE"
    continue
  fi

  if [ "$ue_absent_since" -gt 0 ]; then
    gone=$(( now - ue_absent_since ))
    if [ "$gone" -ge "$UE_DEAD_S" ]; then
      fire "UE-DEAD ${gone}s"
    fi
  fi
done
