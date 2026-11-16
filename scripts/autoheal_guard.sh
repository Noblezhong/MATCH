#!/usr/bin/env bash
ROOT_DIR="${MATCH_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# Cron guard: resurrect the autoheal tmux session if it was killed
# (e.g. collateral damage of an OOM kill, as happened 2026-09-28).
TMUX=$(command -v tmux || echo /usr/bin/tmux)
if ! "$TMUX" has-session -t autoheal 2>/dev/null; then
  echo "$(date "+%F %T") autoheal session missing -> resurrecting" >> ${ROOT_DIR}/table5_logs/autoheal_guard.log
  "$TMUX" new-session -d -s autoheal -c ${ROOT_DIR} "bash ${ROOT_DIR}/scripts/autoheal_table5.sh"
fi
