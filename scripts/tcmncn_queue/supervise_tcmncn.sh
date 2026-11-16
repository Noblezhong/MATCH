#!/usr/bin/env bash
# Keep the TCM+NCN eval alive until 1418 episode dirs are on disk.
# The evaluator resumes: episode dirs already present are skipped, so a relaunch
# costs nothing but the startup.
ROOT_DIR="${MATCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
D="${TCMNCN_RESULT_DIR:-$ROOT_DIR/eval_continuous_tcmncn_seen1418_20260918_fast_x10}"
TARGET=1418

count_dirs() {
  find "$D" -maxdepth 1 -mindepth 1 -type d ! -name profile_logs 2>/dev/null | wc -l
}

while true; do
  n=$(count_dirs)
  if [ "$n" -ge "$TARGET" ]; then
    echo "[$(date '+%F %T')] DONE $n/$TARGET — supervisor exiting"
    break
  fi
  echo "[$(date '+%F %T')] launching TCM+NCN eval (have $n/$TARGET)"
  bash "$ROOT_DIR/scripts/tcmncn_queue/run_tcmncn_full.sh"
  rc=$?
  echo "[$(date '+%F %T')] eval exited rc=$rc (have $(count_dirs)/$TARGET); relaunch in 20s"
  sleep 20
done
