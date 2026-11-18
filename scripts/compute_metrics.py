#!/usr/bin/env python3
"""Unified MATCH evaluation metrics from evaluator JSONL logs.

Single entry point for the main-table 9 metrics across all evaluation
paradigms (TC-ON / TC-OFF, PPO, Continuous, Stop-go, Rule-based), with a
single, verified aggregation protocol:

  SR / OSR / NE / control steps / logical E2E
      from episode_end records (identical for every paradigm)
  CR   versioned episode_end.cr_collision for new runs; legacy logs use
       episode_end.collision AND any logged frame has_collided
       (scans <eval_dir>/<episode>/log/*.json). This is the operational
       definition used in Tables 1 and 3: the terminal flag also covers
       proximity/stall/timeout outcomes, while AirSim collision events can
       appear at reset or be consumed before a frame is logged. Do not
       interpret this conjunction as an exact physical-contact rate.
  time_shift / state_shift   paradigm-specific application-time sampling
  buffer_wait                paradigm-specific stalled-time accounting

Usage:
  python compute_metrics.py --input <eval.jsonl> [--eval-dir <dir>]
      [--output <out.json>] [--method auto|tc|ppo|continuous|stopgo|rule]
      [--wait-denominator all|waiting]

--eval-dir defaults to the JSONL's parent's parent (evaluations/<run>/).
--wait-denominator: 'all' (default) averages buffer_wait_s over every
episode — matches the main-table PPO / rule / continuous / stop-go rows;
'waiting' averages only over episodes that experienced any wait — matches
the two TC rows (TC-OFF 19.48, TC-ON 92.81).

Paradigm auto-detection (first 500 valid records):
  record_type trajectory_result  -> tc
  record_type scheduler_step     -> ppo
  record_type decision_stop      -> rule
  record_type execution          -> continuous
  otherwise                      -> stopgo
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

EPSILON_MS = 1e-6

# ---------------------------------------------------------------- utils


def _fmean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _number(record: dict, key: str, default=None) -> float | None:
    value = record.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _episode_id(record: dict) -> str | None:
    names = record.get("seq_names") or []
    return str(names[0]) if names else None


def _jsonl_records(path: Path):
    """Yield (line_number, record_or_None) for each non-empty line."""
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_number, json.loads(line)
            except json.JSONDecodeError:
                yield line_number, None


# ------------------------------------------------------------- detection


def detect_method(path: Path, invalid: list[int]) -> str:
    kinds: dict[str, int] = defaultdict(int)
    sample = 0
    for line_number, record in _jsonl_records(path):
        if record is None:
            invalid.append(line_number)
            continue
        sample += 1
        if sample > 500:
            break
        kind = record.get("record_type")
        if kind:
            kinds[kind] += 1
        if record.get("scheduler_action") is not None:
            kinds["<scheduler_action>"] += 1
    if kinds.get("trajectory_result"):
        return "tc"
    if kinds.get("scheduler_step"):
        return "ppo"
    if kinds.get("decision_stop") or kinds.get("<scheduler_action>"):
        return "rule"
    if kinds.get("execution"):
        return "continuous"
    return "stopgo"


# --------------------------------------------------- paradigm extractors
# Each returns dict[episode_id, record] of terminals plus per-episode
# application shift events and wait accumulation, in a uniform dict form.


def _collect_common(path: Path, invalid: list[int]):
    """One pass: terminals, plus a raw per-episode event bag.

    Returns (terminals, events) where events[ep] is a dict with
    'shifts' (list of (time_s, state_m) tuples) and 'wait_ms'.
    The wait_ms accumulation rule is paradigm-specific and supplied by the
    caller through 'wait_hooks'.
    """
    raise NotImplementedError


# We keep the per-paradigm passes explicit instead of over-abstracting:
# each evaluation line writes slightly different fields, and the protocol
# differences (which records sample the shift, what counts as waiting)
# must stay visible.

# ------------------------------------------------------------ TC line


def extract_tc(path: Path, invalid: list[int]) -> tuple[dict, dict]:
    terminals: dict[str, dict] = {}
    shifts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    buffer_wait_ms: dict[str, float] = defaultdict(float)
    pending_waits: dict[str, list[float]] = {}
    records = 0

    for line_number, record in _jsonl_records(path):
        if record is None:
            invalid.append(line_number)
            continue
        records += 1
        ep = _episode_id(record)
        if ep is None:
            continue
        kind = record.get("record_type")
        if kind == "episode_end":
            if ep in terminals:
                raise ValueError(f"duplicate terminal episode {ep}")
            terminals[ep] = record
            pending_waits.pop(ep, None)
        elif kind == "trajectory_result":
            time_ms = _number(record, "time_shift_at_apply_ms")
            state_m = _number(record, "state_shift_at_apply_m")
            if time_ms is not None and state_m is not None:
                if time_ms < 0.0 or state_m < 0.0:
                    raise ValueError(f"negative shift in episode {ep}: {time_ms}, {state_m}")
                shifts[ep].append((time_ms / 1000.0, state_m))
            # Corrected definition (2026-08-07): pure stop-and-wait duration.
            # buffer_exhausted records mark the start of a wait (t0); the
            # matching trajectory_result's result_ready_logical_ms marks the
            # instant the awaited result became ready (t1). Wait time =
            # t1 - t0. The earlier definition accumulated the logical
            # interval between consecutive buffer_wait records, which also
            # contained the flight time between two waits and inflated the
            # metric ~26x.
            pending = pending_waits.get(ep)
            if pending:
                t0 = pending.pop(0)
                ready = _number(
                    record, "result_ready_logical_ms",
                    _number(record, "logical_elapsed_ms", 0.0),
                )
                if ready >= t0:
                    buffer_wait_ms[ep] += ready - t0
        elif kind == "buffer_wait":
            if record.get("buffer_exhausted"):
                now = _number(record, "logical_elapsed_ms", 0.0)
                pending_waits.setdefault(ep, []).append(now)

    events = {
        ep: {"shifts": shifts[ep], "wait_ms": buffer_wait_ms[ep]}
        for ep in terminals
    }
    return terminals, events, records


# --------------------------------------------------------- PPO line


def extract_ppo(path: Path, invalid: list[int]) -> tuple[dict, dict, int]:
    terminals: dict[str, dict] = {}
    waiting_ms: dict[str, float] = defaultdict(float)
    shifts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    elapsed_ms: dict[str, float] = defaultdict(float)
    records = 0

    for line_number, record in _jsonl_records(path):
        if record is None:
            invalid.append(line_number)
            continue
        records += 1
        kind = record.get("record_type")
        if kind not in {"decision", "scheduler_step", "episode_end"}:
            continue
        ep = _episode_id(record)
        if ep is None:
            continue
        if kind == "episode_end":
            if ep in terminals:
                raise ValueError(f"duplicate terminal episode {ep}")
            terminals[ep] = record
            continue

        elapsed = _number(record, "elapsed_ms")
        if elapsed is None:
            continue
        action_ms = _number(record, "airsim_action_latency_ms", 0.0)
        if action_ms > elapsed + EPSILON_MS:
            raise ValueError(f"action exceeds elapsed time at line {line_number}")
        elapsed_ms[ep] += elapsed
        waiting_ms[ep] += max(0.0, elapsed - action_ms)

        if kind == "decision":
            time_ms = _number(record, "time_drift_ms")
            state_m = _number(record, "state_drift_m")
            if time_ms is not None and state_m is not None:
                shifts[ep].append((time_ms / 1000.0, state_m))
            continue

        # scheduler_step: sample shift at actual application points.
        #
        # Shift scope (user decision 2026-09-17): ONLY the AHS scheduling
        # loop guidance applications count toward time/state shift.
        # Module-internal events are excluded because they do NOT measure
        # misalignment of an applied result:
        #   - TCM exec-time corrections (trajectory_mode == "corrected"):
        #     the recorded drift is the staleness of the REPLACED guidance,
        #     i.e. the correction trigger condition (>= delta_cor by design),
        #     not the correction own observation age.
        #   - NCN-controlled steps (control_source in {"ncn",
        #     "edge_recovery"}): applications only happen at recovery
        #     landing, and their age includes the NCN hold/outage window,
        #     not the guidance own latency. NOTE: recovery landings are
        #     tagged control_source == "edge_recovery" (not "ncn") on the
        #     step that applies the recovery result (ncn_active=True,
        #     ncn_recovery_reason set).
        # AGENT REMINDER: keep this exclusion when editing extract_ppo, and
        # keep the markers set at the source (trajcorr_apply.py writes
        # trajectory_mode on both TCM paths; match_eval.py _log_ncn_event
        # writes control_source). Dropping it silently inflates MATCH rows:
        # 2026-09-16 run, 1418 eps: 5.67s/2.05m with module events vs
        # 3.65s/0.04m without (verified both ways; TCM 2212 + NCN 6118
        # events excluded, all other metrics unaffected).
        if record.get("trajectory_switch_applied_before_action") and record.get("trajectory_mode") != "corrected":
            time_ms = _number(record, "time_drift_before_ms")
            state_m = _number(record, "state_drift_before_m")
            if time_ms is not None and state_m is not None:
                shifts[ep].append((time_ms / 1000.0, state_m))
        applied_ms = record.get("result_applied_logical_ms")
        if (
            isinstance(applied_ms, (int, float))
            and math.isclose(float(applied_ms), float(record.get("logical_elapsed_ms", 0.0)), abs_tol=EPSILON_MS)
            and record.get("control_source") not in ("ncn", "edge_recovery")
        ):
            time_ms = _number(record, "time_drift_ms")
            state_m = _number(record, "state_drift_m")
            if time_ms is not None and state_m is not None:
                shifts[ep].append((time_ms / 1000.0, state_m))

    events = {ep: {"shifts": shifts[ep], "wait_ms": waiting_ms[ep]} for ep in terminals}
    return terminals, events, records


# ----------------------------------------------------- Continuous / Stop-go


def extract_action_line(path: Path, invalid: list[int], method: str, wait_mode: str) -> tuple[dict, dict, int]:
    """Continuous and Stop-go.

    Wait definitions (verified against the main table):
      e2e-action (stop-go, ppo): terminal E2E minus executed action time.
      approved (continuous): sum of non-moving local observation, non-moving
        DINO, and edge-result waiting components, per episode.
    """
    terminals: dict[str, dict] = {}
    shifts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    action_ms: dict[str, float] = defaultdict(float)
    wait_parts: dict[str, list[float]] = defaultdict(list)  # approved components
    records = 0

    for line_number, record in _jsonl_records(path):
        if record is None:
            invalid.append(line_number)
            continue
        records += 1
        ep = _episode_id(record)
        if ep is None:
            continue
        kind = record.get("record_type")

        if kind == "episode_end":
            if ep in terminals:
                raise ValueError(f"duplicate terminal episode {ep}")
            terminals[ep] = record
            continue

        if method == "continuous":
            if kind == "decision":
                time_ms = _number(record, "action_age_ms")
                state_m = _number(record, "state_drift_m")
                if time_ms is not None and state_m is not None:
                    shifts[ep].append((time_ms / 1000.0, state_m))
            elif kind == "execution":
                action_ms[ep] += _number(record, "action_sim_time_ms", 0.0)
                if wait_mode == "approved":
                    wait_parts[ep].append(_number(record, "execution_post_obs_latency_ms", 0.0))
                    wait_parts[ep].append(_number(record, "execution_post_dino_latency_ms", 0.0))
            if wait_mode == "approved" and kind in ("decision", "decision_stop"):
                # A single hover event is written to both the decision and the
                # execution record of the same loop iteration, carrying the
                # identical value. Count it once, on the decision side only.
                wait_parts[ep].append(_number(record, "hover_wait_ms", 0.0))
        else:  # stopgo: every non-terminal record is an executed step
            time_ms = _number(record, "action_age_ms")
            state_m = _number(record, "state_drift_m")
            if time_ms is not None and state_m is not None:
                shifts[ep].append((time_ms / 1000.0, state_m))
            action_ms[ep] += _number(record, "action_sim_time_ms", 0.0)

    events: dict[str, dict] = {}
    for ep in terminals:
        if method == "continuous" and wait_mode == "approved":
            wait = sum(wait_parts[ep])
        else:
            wait = terminals[ep].get("logical_elapsed_ms", 0.0) - action_ms[ep]
        if wait < -1e-6:
            raise ValueError(f"negative waiting time in episode {ep}: {wait}ms")
        events[ep] = {"shifts": shifts[ep], "wait_ms": max(0.0, wait)}
    return terminals, events, records


# ------------------------------------------------------------ Rule-based


def extract_rule(path: Path, invalid: list[int]) -> tuple[dict, dict, int]:
    terminals: dict[str, dict] = {}
    shifts: dict[str, list[tuple[float, float]]] = defaultdict(list)
    hover_wait_ms: dict[str, float] = defaultdict(float)
    records = 0

    for line_number, record in _jsonl_records(path):
        if record is None:
            invalid.append(line_number)
            continue
        records += 1
        ep = _episode_id(record)
        if ep is None:
            continue
        kind = record.get("record_type")
        if kind == "episode_end":
            if ep in terminals:
                raise ValueError(f"duplicate terminal episode {ep}")
            terminals[ep] = record
            continue
        # A single hover event is written to both the decision and the
        # execution record of the same loop iteration, carrying the identical
        # value. Summing both would count the same wait twice; count it once,
        # on the decision side only.
        if kind in ("decision", "decision_stop"):
            hover = _number(record, "hover_wait_ms", 0.0)
            if hover > 0:
                hover_wait_ms[ep] += hover
        if kind == "decision":
            time_ms = _number(record, "action_age_ms")
            state_m = _number(record, "state_drift_m")
            if time_ms is not None and state_m is not None:
                shifts[ep].append((time_ms / 1000.0, state_m))

    events = {
        ep: {"shifts": shifts[ep], "wait_ms": hover_wait_ms[ep]}
        for ep in terminals
    }
    return terminals, events, records


# ------------------------------------------------------------ collision


def _strip_prefix(name: str) -> str:
    for prefix in ("success_", "oracle_success_", "oracle_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def scan_airsim_collision(eval_dir: Path, episode_ids: set[str]) -> dict[str, bool]:
    """AirSim judgment: has_collided=True in any frame of the episode's log.

    Scans <eval_dir>/<episode-id>/log/*.json (episode dirs may carry a
    success_/oracle_ prefix). Missing dirs/logs -> episode counts as no
    collision and is reported in warnings by the caller.
    """
    result: dict[str, bool] = {}
    if not eval_dir.is_dir():
        return result
    for entry in eval_dir.iterdir():
        if not entry.is_dir():
            continue
        name = _strip_prefix(entry.name)
        if name not in episode_ids or name in result:
            continue
        log_dir = entry / "log"
        if not log_dir.is_dir():
            result[name] = False
            continue
        collided = False
        for frame in sorted(log_dir.glob("*.json")):
            try:
                data = json.loads(frame.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError, OSError):
                continue
            if data.get("sensors", {}).get("state", {}).get("collision", {}).get("has_collided", False):
                collided = True
                break
        result[name] = collided
    return result


# ----------------------------------------------------------- aggregation


def build_result(
    path: Path,
    method: str,
    terminals: dict,
    events: dict,
    records: int,
    invalid: list[int],
    eval_dir: Path,
    wait_denominator: str = "all",
    expected_episodes: int = 1418,
) -> dict:
    episodes = []
    policies = {r.get("collision_policy", "legacy_terminal_and_saved_frame") for r in terminals.values()}
    if len(policies) != 1:
        raise ValueError("mixed collision policies: split old and new evaluation inputs")
    collision_policy = next(iter(policies))
    if collision_policy not in {"legacy_terminal_and_saved_frame", "episode_collision_v1"}:
        raise ValueError(f"unsupported collision policy: {collision_policy}")
    if collision_policy == "episode_collision_v1" and any(type(r.get("cr_collision")) is not bool for r in terminals.values()):
        raise ValueError("new collision policy requires boolean cr_collision for every episode")
    airsim = scan_airsim_collision(eval_dir, set(terminals)) if collision_policy == "legacy_terminal_and_saved_frame" else {}
    for ep in sorted(terminals):
        terminal = terminals[ep]
        ev = events.get(ep, {"shifts": [], "wait_ms": 0.0})
        shifts = ev["shifts"]
        # New runs use the shared episode collision policy; legacy evidence
        # fields remain visible for historical report compatibility.
        has_airsim = bool(airsim.get(ep, False))
        episodes.append(
            {
                "episode_id": ep,
                "success": int(bool(terminal.get("success"))),
                "oracle_success": int(bool(terminal.get("oracle_success"))),
                "collision": int(terminal["cr_collision"]) if collision_policy == "episode_collision_v1" else int(bool(terminal.get("collision")) and has_airsim),
                "episode_end_collision": int(bool(terminal.get("collision"))),
                "airsim_collision": int(bool(terminal.get("cr_airsim_collision"))) if collision_policy == "episode_collision_v1" else int(has_airsim),
                "navigation_error": _number(terminal, "final_ne_m"),
                "e2e_latency_s": (
                    _number(terminal, "logical_elapsed_ms") / 1000.0
                    if _number(terminal, "logical_elapsed_ms") is not None else None
                ),
                "executed_waypoints": int(terminal.get("control_steps") or 0),
                "time_shift_s": _fmean([t for t, _ in shifts]),
                "state_shift_m": _fmean([s for _, s in shifts]),
                "applied_guidance": len(shifts),
                "buffer_wait_s": ev["wait_ms"] / 1000.0,
            }
        )

    n = len(episodes)
    if n == 0:
        raise ValueError("no terminal episodes found")
    summary = {
        "collision_policy": collision_policy,
        "SR": sum(e["success"] for e in episodes) / n,
        "OSR": sum(e["oracle_success"] for e in episodes) / n,
        "CR": sum(e["collision"] for e in episodes) / n,
        "NE_m": statistics.fmean(e["navigation_error"] for e in episodes if e["navigation_error"] is not None),
        "avg_control_steps": statistics.fmean(e["executed_waypoints"] for e in episodes),
        "mean_logical_E2E_s": statistics.fmean(e["e2e_latency_s"] for e in episodes if e["e2e_latency_s"] is not None),
        "NE_count": sum(e["navigation_error"] is not None for e in episodes),
        "E2E_count": sum(e["e2e_latency_s"] is not None for e in episodes),
        "time_shift_s": _fmean([e["time_shift_s"] for e in episodes if e["time_shift_s"] is not None]),
        "state_shift_m": _fmean([e["state_shift_m"] for e in episodes if e["state_shift_m"] is not None]),
        # Self-describing scope marker: shift excludes TCM-trigger and NCN
        # module-internal events (see extract_ppo comment, 2026-09-17).
        "shift_scope": "applied_edge_guidance_excluding_module_internal" if method == "ppo" else "applied_edge_guidance",
        # Main table rows PPO/rule/continuous/stop-go average over all episodes;
        # the two TC rows (TC-OFF 19.48, TC-ON 92.81) average only over episodes
        # that experienced any wait. --wait-denominator reproduces either.
        "buffer_wait_s": statistics.fmean(
            [e["buffer_wait_s"] for e in episodes if wait_denominator == "all" or e["buffer_wait_s"] > 0.0]
        ),
    }
    warnings: list[str] = []
    if n != expected_episodes:
        warnings.append(f"expected {expected_episodes} terminal episodes, got {n}")
    missing_terminal = [e["episode_id"] for e in episodes if e["navigation_error"] is None or e["e2e_latency_s"] is None]
    if missing_terminal:
        warnings.append(f"{len(missing_terminal)} episodes with missing terminal NE/E2E: {missing_terminal[:10]}")
    if invalid:
        warnings.append(f"{len(invalid)} invalid JSON lines: {invalid[:10]}")
    missing_shifts = [e["episode_id"] for e in episodes if e["time_shift_s"] is None]
    if missing_shifts and method != "stopgo":
        warnings.append(f"{len(missing_shifts)} episodes without shift samples")
    unscanned = [e["episode_id"] for e in episodes if e["episode_id"] not in airsim]
    if unscanned and collision_policy == "legacy_terminal_and_saved_frame":
        warnings.append(f"{len(unscanned)} episodes without AirSim frame scan (CR counted as no collision)")

    return {
        "method": method,
        "input": str(path),
        "records": records,
        "episodes": n,
        "warnings": warnings,
        "summary": summary,
        "episode_details": episodes,
    }


# ------------------------------------------------------------------ main


EXTRACTORS = {
    "tc": extract_tc,
    "ppo": extract_ppo,
    "continuous": extract_action_line,
    "stopgo": extract_action_line,
    "rule": extract_rule,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path,
                        help="one JSONL file, a profile_logs directory, or an evaluation directory")
    parser.add_argument("--eval-dir", type=Path,
                        help="evaluation run directory containing episode log dirs "
                             "(default: parent of the JSONL's parent)")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--method", default="auto",
                        choices=("auto",) + tuple(EXTRACTORS))
    parser.add_argument("--wait-mode", default="auto", choices=("auto", "e2e-action", "approved"),
                        help="buffer-wait definition: 'auto' uses the main-table "
                             "definition per paradigm (stop-go/ppo: E2E minus action "
                             "time; continuous: approved component sum; rule: hover "
                             "sum; tc: buffer_wait records)")
    parser.add_argument("--wait-denominator", default="all", choices=("all", "waiting"),
                        help="buffer_wait_s averaging denominator: 'all' (default, "
                             "matches PPO/rule/continuous/stop-go rows) averages over "
                             "every episode; 'waiting' averages only over episodes "
                             "with any wait (matches the two TC main-table rows)")
    parser.add_argument("--exclude-episode", action="append", default=[],
                        help="episode ID to exclude from every metric (repeatable)")
    parser.add_argument("--expected-episodes", type=int,
                        help="expected final episode count; defaults to 1418 minus exclusions")
    args = parser.parse_args()

    if args.input.is_file():
        inputs = [args.input]
        eval_dir = args.eval_dir or args.input.parent.parent
    elif args.input.is_dir():
        profile_dir = args.input / "profile_logs" if (args.input / "profile_logs").is_dir() else args.input
        inputs = sorted(profile_dir.glob("*.jsonl"))
        eval_dir = args.eval_dir or (args.input if profile_dir != args.input else args.input.parent)
    else:
        parser.error(f"input not found: {args.input}")
    if not inputs:
        parser.error(f"no JSONL inputs under {args.input}")

    invalid: list[int] = []
    method = args.method if args.method != "auto" else detect_method(inputs[0], invalid)
    if args.method == "auto" and invalid:
        # detection consumed some invalid lines; re-run with a fresh list
        invalid = []
    wait_mode = args.wait_mode
    if wait_mode == "auto":
        # Main-table definitions per paradigm
        wait_mode = "approved" if method == "continuous" else "e2e-action"

    terminals: dict = {}
    events: dict = {}
    records = 0
    duplicate_terminals = 0
    input_collision_policies = set()
    for input_path in inputs:
        if method == "continuous":
            file_terminals, file_events, file_records = extract_action_line(input_path, invalid, "continuous", wait_mode)
        elif method == "stopgo":
            file_terminals, file_events, file_records = extract_action_line(input_path, invalid, "stopgo", wait_mode)
        else:
            file_terminals, file_events, file_records = EXTRACTORS[method](input_path, invalid)
        input_collision_policies.update(r.get("collision_policy", "legacy_terminal_and_saved_frame") for r in file_terminals.values())
        if len(input_collision_policies) > 1:
            raise ValueError("mixed collision policies across input segments")
        records += file_records
        duplicate_terminals += len(set(terminals).intersection(file_terminals))
        terminals.update(file_terminals)
        events.update(file_events)

    excluded = set(args.exclude_episode)
    absent = excluded.difference(terminals)
    if absent:
        parser.error(f"excluded episode IDs absent from terminals: {sorted(absent)}")
    for episode_id in excluded:
        terminals.pop(episode_id)
        events.pop(episode_id, None)

    result = build_result(args.input, method, terminals, events, records, invalid, eval_dir,
                          wait_denominator=args.wait_denominator,
                          expected_episodes=args.expected_episodes if args.expected_episodes is not None else 1418-len(excluded))
    result["inputs"] = [str(path) for path in inputs]
    result["excluded_episode_ids"] = sorted(excluded)
    result["duplicate_terminals_resolved_by_latest_segment"] = duplicate_terminals
    output = args.output or (args.input.with_name(args.input.stem + "_metrics.json") if args.input.is_file()
                             else eval_dir / "combined_metrics.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(
        {"method": result["method"], "episodes": result["episodes"],
         "records": result["records"], "warnings": result["warnings"],
         "summary": result["summary"]},
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
