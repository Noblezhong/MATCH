#!/usr/bin/env python3
"""Compare per-episode metrics between two eval run dirs (regression check)."""
import json, sys
from pathlib import Path

SCALAR = ["terminal_reason", "success", "oracle_success", "collision",
          "control_steps", "scheduler_steps", "final_ne_m",
          "tcm_corrected_applies", "tcm_mid_exec_corrections", "tcm_stale_dropped",
          "ncn_actions", "ncn_takeovers", "ncn_stale_results", "ncn_recoveries",
          "logical_elapsed_ms"]
NESTED = ["tcm_lock_completions"]


def episodes(d):
    out = {}
    for f in sorted(Path(d, "profile_logs").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("record_type") != "episode_end":
                continue
            seq = r.get("seq_names")
            seq = seq[0] if isinstance(seq, list) and seq else str(seq)
            out[seq] = r
    return out


def eq(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return round(a, 6) == round(b, 6)
    return a == b


a, b = episodes(sys.argv[1]), episodes(sys.argv[2])
print(f"{sys.argv[1]}: {len(a)} episodes; {sys.argv[2]}: {len(b)} episodes")
assert set(a) == set(b), f"episode sets differ: {set(a) ^ set(b)}"
diffs = 0
for seq in sorted(a):
    for k in SCALAR + NESTED:
        va, vb = a[seq].get(k), b[seq].get(k)
        if not eq(va, vb):
            diffs += 1
            print(f"DIFF {seq[:12]} {k}: {va!r} vs {vb!r}")
print("IDENTICAL" if diffs == 0 else f"{diffs} field diffs")
sys.exit(1 if diffs else 0)
