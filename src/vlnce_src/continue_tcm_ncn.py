"""Continuous + TCM + NCN paradigm evaluator (ablation cell: Continuous w=5 + TCM + NCN).

NEW FILE — the no-AHS cell of Table 3.  ``match_eval.py`` is the only other
evaluator carrying both modules, but its base is the AHS hybrid scheduler, so it
cannot produce the no-AHS variant.  This file forks ``continue_ncn.py``
(the audited Continuous+NCN cell) and adds the TCM branch at the same five
integration points used by ``continue_tcm.py``.  Neither parent file is
modified.

Fidelity contract:
  - ``--trajcorr_mode off``: behavior identical to ``continue_ncn.py``
    (the audited Continuous+NCN row).
  - ``--trajcorr_mode on``: Continuous + TCM + NCN.

Interaction between the two modules (mirrors ``match_eval.py``, where the
target lock is routed with priority over everything else):
  - While the TCM target lock is active the NCN local-action gate is suppressed
    and no new edge request is issued; the locked buffer is flown by the normal
    ``makeActionsChunk`` path.
  - A correction installs a fresh ``refined_waypoints`` buffer into the same
    slot the edge waypoints use, so both modules share one execution path.
  - An edge result that lands during the lock is dropped as stale (never
    applied, never re-locked) — the NCN interval bookkeeping is left untouched
    on that path.

record_type vocabulary is unchanged: {decision, execution, decision_stop,
episode_end}.  TCM fields are extra keys only (compute_metrics auto-detection
stays continuous).

Keep in sync with ``continue_ncn.py``: if its loop changes, the loop below
must be re-checked.
"""

import copy
import json
import math
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import tqdm

sys.path.append(str(Path(str(os.getcwd())).resolve()))
from src.vlnce_src.continuous_runtime import check_evaluation, FixedStepEdgePlanner as LatestOnlyEdgePlanner, run_decision_cycle as _run_decision_cycle, request_due

from assist import Assist
from src.common.param import args, data_args, model_args
from src.model_wrapper.profile_travel_llm import ProfileTravelModelWrapper
from src.vlnce_src.closeloop_util import (
    BatchIterator,
    CheckPort,
    initialize_env_eval,
    is_dist_avail_and_initialized,
    save_to_dataset_eval,
    setup,
)
from src.vlnce_src.comm_delay import (
    BandwidthTrace,
    calculate_latency_ms,
    default_trace_path,
    estimate_uplink_payload_bits_from_episodes,
)
from src.vlnce_src.dino_monitor_online import DinoMonitor
from src.vlnce_src.fast_eval_time import (
    FastEvalClock,
    FastResultTiming,
    action_timing,
    configure_fast_eval_output,
)
from src.vlnce_src.aerodpo_eval_contract import resolve_aerodpo_stop
from src.vlnce_src.eval_contract import check_collision_without_tiny_diff
from src.vlnce_src.ncn_runtime import EdgeLatencyEstimate, NCNRuntime
from src.vlnce_src.trajcorr_apply import TcmRuntime, TrajcorrMixin
from env_uav import AirVLNENV
from utils.logger import logger
from utils.utils import *


class MatchContinuousTravelModelWrapper(TrajcorrMixin, ProfileTravelModelWrapper):
    """Continuous wrapper with TCM trajectory regeneration added.

    The mixin only contributes ``regenerate_trajectory``; everything else is
    inherited unchanged from ``ProfileTravelModelWrapper``.
    """


@dataclass
class Snapshot:
    request_id: int
    submitted_step: int
    episode: List[Dict[str, Any]]
    target_position: List[float]
    object_info: str
    assist_notice: Optional[str]
    observation_timestamp: int
    observation_pose: List[float]
    submitted_perf_time: float
    submitted_wall_time: float
    obs_latency_ms: float
    groundingdino_latency_ms: float
    submitted_logical_ms: Optional[float] = None
    planner_started_logical_ms: Optional[float] = None


@dataclass
class PlannerResult:
    request_id: int
    submitted_step: int
    observation_timestamp: int
    observation_pose: List[float]
    submitted_perf_time: float
    submitted_wall_time: float
    ready_wall_time: float
    obs_latency_ms: float
    groundingdino_latency_ms: float
    llm_latency_ms: float
    traj_latency_ms: float
    uplink_payload_bits: int
    uplink_payload_mb: float
    uplink_bandwidth_mbps: float
    uplink_latency_ms: float
    llm_output: List[List[float]]
    refined_waypoints: List[List[float]]
    submitted_logical_ms: Optional[float] = None
    edge_arrival_logical_ms: Optional[float] = None
    ready_logical_ms: Optional[float] = None
    # TCM correction inputs, restored by the planner with the original TC-ON
    # rotation chain (continue_eval.py carries these two fields; the NCN fork
    # dropped them, which would silently disable every correction with a
    # ``missing_coarse_frame`` fallback).
    coarse_local: Optional[List[float]] = None
    coarse_goal_world: Optional[List[float]] = None


def _tcm_apply_result(tcm: TcmRuntime, state, eval_env, model_wrapper, clock, applied_result):
    """TCM-aware apply; returns (effective_result, tcm_extra) or (None, tcm_extra) on stale-drop."""
    return tcm.apply_result(
        state=state,
        eval_env=eval_env,
        model_wrapper=model_wrapper,
        clock=clock,
        result=applied_result,
        record_request_ne=True,
    )


def _metric_summary(values):
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _write_jsonl_line(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _set_console_log_message_only():
    plain_formatter = logging.Formatter("%(message)s")
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(plain_formatter)


def _print_episode_header(episode_idx, env_batch, chunk_waypoints):
    logger.info(
        f"\n[episode {episode_idx:04d}] seq={env_batch['seq_name']} "
        f"map={env_batch['map_name']} chunk={int(chunk_waypoints)}"
    )


def _fmt_point(point):
    arr = np.asarray(point, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{v:.2f}" for v in arr[:3]) + "]"


def _point_delta(a, b):
    a_arr = np.asarray(a, dtype=np.float64).reshape(-1)[:3]
    b_arr = np.asarray(b, dtype=np.float64).reshape(-1)[:3]
    return float(np.linalg.norm(a_arr - b_arr))


def _fmt_optional(value, fmt):
    if value is None:
        return "None"
    return format(value, fmt)


def _fmt_optional_ms(value, fmt=".1f"):
    if value is None:
        return "None"
    return f"{format(value, fmt)}ms"


def _fmt_optional_m(value, fmt=".2f"):
    if value is None:
        return "None"
    return f"{format(value, fmt)}m"


def _fmt_optional_mbps(value, fmt=".2f"):
    if value is None:
        return "None"
    return f"{format(value, fmt)}Mbps"


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(value)


def _waypoint_segment_stats(points):
    if points is None:
        return {"count": 0, "span_m": None, "step_mean_m": None, "step_min_m": None, "step_max_m": None, "step_list": []}
    arr = np.asarray(points, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "span_m": None, "step_mean_m": None, "step_min_m": None, "step_max_m": None, "step_list": []}
    arr = arr.reshape(-1, 3)
    if len(arr) == 1:
        return {"count": 1, "span_m": 0.0, "step_mean_m": 0.0, "step_min_m": 0.0, "step_max_m": 0.0, "step_list": []}
    segments = np.linalg.norm(np.diff(arr[:, :3], axis=0), axis=1)
    return {
        "count": int(len(arr)),
        "span_m": float(np.linalg.norm(arr[-1, :3] - arr[0, :3])),
        "step_mean_m": float(segments.mean()),
        "step_min_m": float(segments.min()),
        "step_max_m": float(segments.max()),
        "step_list": segments.tolist(),
    }


def _print_trajectory_bundle(episode_idx, decision_step, llm_output, refined_waypoints, current_pose=None):
    coarse_summary = _fmt_point(llm_output[0]) if len(llm_output) > 0 else "[]"
    refined_summary = _waypoint_segment_stats(refined_waypoints)
    final_waypoint = _fmt_point(refined_waypoints[-1]) if len(refined_waypoints) > 0 else "[]"
    line = (
        f"[ep {episode_idx:04d} decision_step={decision_step} plan] "
        f"coarse_local={coarse_summary} "
    )
    if current_pose is not None:
        line += f" | uav_world={_fmt_point(current_pose)}"
    line += (
        f" | final_world={final_waypoint}"
        f" | wp_n={refined_summary['count']}"
        f" | span={_fmt_optional_m(refined_summary['span_m'])}"
    )
    logger.info(line)


def _print_exec_profile_line(episode_idx, step_idx, profile_info, current_chunk=None, current_pose=None):
    line = (
        f"[ep {episode_idx:04d} exec_step={step_idx}] "
        f"act={profile_info['airsim_action_latency_ms']:.1f}ms "
    )
    if current_pose is not None:
        line += f" | uav_world={_fmt_point(current_pose)}"
    if current_chunk is not None and len(current_chunk) > 0:
        line += (
            f" | exec_world={_fmt_point(current_chunk[0])}"
            f" | exec_dist={_fmt_optional_m(profile_info.get('exec_target_distance_m'))}"
        )
    logger.info(line)


def _print_decision_profile_line(episode_idx, decision_step, profile_info):
    logger.info(
        f"[ep {episode_idx:04d} decision_step={decision_step} timing] "
        f"obs={profile_info['obs_latency_ms']:.1f}ms "
        f"dino={profile_info['groundingdino_latency_ms']:.1f}ms "
        f"bw={_fmt_optional_mbps(profile_info.get('uplink_bandwidth_mbps'))} "
        f"uplink={_fmt_optional_ms(profile_info.get('uplink_latency_ms'))} "
        f"llm={_fmt_optional_ms(profile_info.get('llm_latency_ms'))} "
        f"traj={_fmt_optional_ms(profile_info.get('traj_latency_ms'))} "
        f"total={_fmt_optional_ms(profile_info.get('decision_latency_ms'))} "
        f"stop={int(profile_info.get('predict_done', False))} "
        f"stop_delay={_fmt_optional_ms(profile_info.get('stop_delay_ms'))} "
        f"stop_drift={_fmt_optional_m(profile_info.get('stop_drift_m'))}"
    )


def _build_planner_decision_record(
    env_batch,
    applied_result: PlannerResult,
    current_pose,
    enable_comm_delay: bool,
    chunk_waypoints: int,
    decision_step: int,
    clock: Optional[FastEvalClock] = None,
    applied_exec_step: Optional[int] = None,
):
    if clock is not None and clock.enabled:
        applied_result.applied_logical_ms = clock.now_ms
        applied_result.applied_exec_step = applied_exec_step
    decision_latency_ms = (
        float(applied_result.obs_latency_ms)
        + float(applied_result.groundingdino_latency_ms)
        + float(applied_result.uplink_latency_ms)
        + float(applied_result.llm_latency_ms)
        + float(applied_result.traj_latency_ms)
    )
    action_age_ms = (
        clock.age_ms(applied_result.submitted_logical_ms, applied_result.submitted_perf_time)
        if clock is not None
        else float((time.perf_counter() - applied_result.submitted_perf_time) * 1000.0)
    )
    record = {
        "record_type": "decision",
        "decision_step": int(decision_step),
        "batch_size": 1,
        "seq_names": [env_batch["seq_name"]],
        "map_names": [env_batch["map_name"]],
        "chunk_waypoints": int(chunk_waypoints),
        "comm_delay_enabled": bool(enable_comm_delay),
        "request_id": int(applied_result.request_id),
        "request_submitted_step": int(applied_result.submitted_step),
        "submitted_after_exec_step": int(applied_result.submitted_step),
        "request_observation_timestamp": int(applied_result.observation_timestamp),
        "request_ready_wall_time": float(applied_result.ready_wall_time),
        "obs_latency_ms": float(applied_result.obs_latency_ms),
        "groundingdino_latency_ms": float(applied_result.groundingdino_latency_ms),
        "uplink_payload_bits": int(applied_result.uplink_payload_bits),
        "uplink_payload_mb": float(applied_result.uplink_payload_mb),
        "uplink_bandwidth_mbps": float(applied_result.uplink_bandwidth_mbps),
        "uplink_latency_ms": float(applied_result.uplink_latency_ms),
        "llm_latency_ms": float(applied_result.llm_latency_ms),
        "traj_latency_ms": float(applied_result.traj_latency_ms),
        "decision_latency_ms": float(decision_latency_ms),
        "airsim_action_latency_ms": 0.0,
        "loop_latency_ms": float(decision_latency_ms),
        "action_age_ms": float(action_age_ms),
        "state_drift_m": float(math.dist(applied_result.observation_pose, current_pose)),
        "trajectory_switch_applied": True,
        "hover_wait_ms": 0.0,
        "llm_output": copy.deepcopy(applied_result.llm_output),
        "refined_waypoints": copy.deepcopy(applied_result.refined_waypoints),
        "current_pose": copy.deepcopy(current_pose),
        "first_refined_gap_m": float(_point_delta(applied_result.refined_waypoints[0], current_pose)) if len(applied_result.refined_waypoints) > 0 else None,
        "predict_done": False,
        "stop_delay_ms": None,
        "stop_drift_m": None,
        "predict_dones": [False],
        "collisions": [False],
        "dones": [False],
        "result_ready_logical_ms": applied_result.ready_logical_ms,
        "result_applied_logical_ms": getattr(applied_result, "applied_logical_ms", None),
        "result_applied_exec_step": applied_exec_step,
    }
    if clock is not None:
        record.update(clock.metadata())
    return record


def _load_object_description():
    with open(args.object_name_json_path, "r") as handle:
        return {item["object_name"]: item["object_desc"] for item in json.load(handle)}


from src.vlnce_src.continue_eval import ContinuousEpisodeState






def _run_ncn_action(
    state: ContinuousEpisodeState,
    eval_env: AirVLNENV,
    runtime: NCNRuntime,
    instruction: str,
    clock: FastEvalClock,
    control_step: int,
    predicted_remaining_ms: float,
    buffer_horizon_ms: float,
    trigger_reason: str,
):
    """Run exactly one local action; edge request/accounting remains untouched."""
    image_start = time.perf_counter()
    observation = eval_env.get_aerodpo_rgb_observations()[0]
    image_fetch_ms = (time.perf_counter() - image_start) * 1000.0
    decision_start = time.perf_counter()
    action, model_stop, model_profile, prompt = runtime.act(
        observation["front_bgr"], observation["down_bgr"], observation["state"],
        state.target_position, instruction,
    )
    image_text_to_action_ms = (time.perf_counter() - decision_start) * 1000.0
    clock.advance_blocking(image_text_to_action_ms)

    action_start = time.perf_counter()
    eval_env.makeAeroDPOActions([action])
    action_times = action_timing(
        eval_env, (time.perf_counter() - action_start) * 1000.0, clock.enabled
    )
    clock.advance_action(action_times["action_sim_time_ms"], action_times["action_wall_time_ms"])
    state.sync_runtime_status_from_sim()

    # Refresh benchmark state for collision/NE bookkeeping only.  NCN does not
    # invoke DINO and does not add an edge upload.
    post_obs_start = time.perf_counter()
    outputs = eval_env.get_obs()
    state.process_env_output(outputs)
    post_obs_ms = (time.perf_counter() - post_obs_start) * 1000.0
    clock.advance_blocking(post_obs_ms)

    land_outcome = resolve_aerodpo_stop(
        model_stop, state._calculate_distance_from_position(state.current_sim_pose())
    )
    if land_outcome == "success":
        state.success = True
        state.dones[0] = True
    elif land_outcome == "early_end":
        state.early_end = True
        state.dones[0] = True

    record = {
        "record_type": "execution",
        "control_source": "ncn",
        "ncn_trigger_reason": trigger_reason,
        "ncn_recovery_reason": None,
        "predicted_edge_remaining_ms": float(predicted_remaining_ms),
        "remaining_buffer_horizon_ms": float(buffer_horizon_ms),
        "exec_step": int(control_step),
        "batch_size": 1,
        "seq_names": [state.env_batch["seq_name"]],
        "map_names": [state.env_batch["map_name"]],
        "chunk_waypoints": 1,
        "executed_waypoints": 1,
        "comm_delay_enabled": bool(args.enable_comm_delay),
        "aerodpo_action": [action],
        "model_stop": [bool(model_stop)],
        "termination_backend": "aerodpo_land",
        "model_output_text": model_profile.get("model_output_text"),
        "ncn_prompt": prompt,
        "ncn_image_fetch_ms": float(image_fetch_ms),
        "image_text_to_action_latency_ms": float(image_text_to_action_ms),
        "model_inference_latency_ms": float(model_profile["model_inference_latency_ms"]),
        "native_bridge_latency_ms": model_profile.get("native_bridge_latency_ms"),
        "native_vision_encode_latency_ms": model_profile.get("native_vision_encode_latency_ms"),
        "native_prefill_latency_ms": model_profile.get("native_prefill_latency_ms"),
        "native_decode_latency_ms": model_profile.get("native_decode_latency_ms"),
        "airsim_action_latency_ms": float(action_times["airsim_action_latency_ms"]),
        "action_wall_time_ms": float(action_times["action_wall_time_ms"]),
        "action_sim_time_ms": float(action_times["action_sim_time_ms"]),
        "obs_latency_ms": float(post_obs_ms),
        "execution_post_obs_latency_ms": float(post_obs_ms),
        "execution_post_dino_latency_ms": 0.0,
        "groundingdino_latency_ms": 0.0,
        "uplink_payload_bits": None,
        "uplink_payload_mb": None,
        "uplink_bandwidth_mbps": None,
        "uplink_latency_ms": None,
        "llm_latency_ms": None,
        "traj_latency_ms": None,
        "loop_latency_ms": float(image_text_to_action_ms + action_times["airsim_action_latency_ms"] + post_obs_ms),
        "decision_latency_ms": float(image_text_to_action_ms),
        "collisions": [bool(state.collisions[0])],
        "predict_dones": [bool(model_stop)],
        "dones": [bool(state.dones[0])],
        "current_pose": state.current_sim_pose(),
    }
    record.update(clock.metadata())
    return record


def eval(
    model_wrapper: ProfileTravelModelWrapper,
    assist: Assist,
    eval_env: AirVLNENV,
    eval_save_dir,
    profile_log_path,
    summary_path,
    bandwidth_trace: BandwidthTrace,
    enable_comm_delay: bool,
    chunk_waypoints: int,
    trajcorr_enabled: bool,
    delta_cor_m: float,
):
    assert int(eval_env.batch_size) == 1, "continuous eval currently supports batchSize=1 only"
    model_wrapper.eval()
    ncn_runtime = NCNRuntime(model_args) if _as_bool(args.enable_ncn) else None
    summary_records = []

    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)
        with open(profile_log_path, "w", encoding="utf-8") as profile_fp:
            episode_idx = 0
            while True:
                env_batchs = eval_env.next_minibatch()
                if env_batchs is None:
                    break
                pbar.update(n=1)
                _print_episode_header(episode_idx, env_batchs[0], chunk_waypoints)

                episode_ok = False
                for retry_i in range(3):
                    planner = None
                    try:
                        bandwidth_trace.reset_for_episode(env_batchs[0]["seq_name"])
                        state = ContinuousEpisodeState(
                            env_batchs[0],
                            eval_env,
                            assist,
                            ignore_tiny_diff=True,
                        )
                        episode_clock = FastEvalClock(bool(args.fast_eval), args.fast_eval_speedup)
                        tcm = TcmRuntime(trajcorr_enabled, delta_cor_m)
                        episode_clock.reset(eval_env.initial_obs_latency_ms)
                        planner = LatestOnlyEdgePlanner(
                            model_wrapper,
                            bandwidth_trace,
                            enable_comm_delay,
                            clock=episode_clock,
                        )
                        planner.bind_state(state, lambda event: _write_jsonl_line(profile_fp, dict(event, seq_names=[env_batchs[0]["seq_name"]], map_names=[env_batchs[0]["map_name"]])))
                        planner.control_step = lambda: control_step
                        request_counter = 0
                        control_step = 0
                        active_traj: List[List[float]] = []
                        active_index = 0
                        active_result: Optional[Any] = None

                        warmup_snapshot = state.build_snapshot(request_counter, control_step)
                        warmup_snapshot.submitted_logical_ms = episode_clock.now_ms if episode_clock.enabled else None
                        submitted = planner.submit(warmup_snapshot)
                        assert submitted, "failed to submit warmup snapshot"
                        warmup_hover_start = time.perf_counter()
                        warmup_logical_start = episode_clock.now_ms
                        warmup_result = planner.wait_result()
                        warmup_hover_ms = (
                            episode_clock.now_ms - warmup_logical_start
                            if episode_clock.enabled
                            else (time.perf_counter() - warmup_hover_start) * 1000.0
                        )
                        warmup_result, warmup_tcm_extra = _tcm_apply_result(
                            tcm, state, eval_env, model_wrapper, episode_clock, warmup_result
                        )
                        active_traj = copy.deepcopy(warmup_result.refined_waypoints)
                        active_index = 0
                        active_result = warmup_result
                        edge_latency = EdgeLatencyEstimate(
                            compute_ms=float(warmup_result.llm_latency_ms) + float(warmup_result.traj_latency_ms),
                            alpha=float(args.ncn_response_ema_alpha),
                        )
                        ncn_actions_in_interval = 0
                        state.record_request_ne()
                        current_pose = state.current_sim_pose()
                        warmup_record = _build_planner_decision_record(
                            env_batchs[0],
                            warmup_result,
                            current_pose,
                            enable_comm_delay,
                            chunk_waypoints,
                            decision_step=0,
                            clock=episode_clock,
                            applied_exec_step=control_step,
                        )
                        warmup_record["hover_wait_ms"] = float(warmup_hover_ms)
                        warmup_record["loop_latency_ms"] = float(warmup_hover_ms)
                        warmup_record["predict_dones"] = [bool(x) for x in state.predict_dones]
                        warmup_record["collisions"] = [bool(x) for x in state.collisions]
                        warmup_record["dones"] = [bool(x) for x in state.dones]
                        warmup_record.update(warmup_tcm_extra)
                        _write_jsonl_line(profile_fp, warmup_record)
                        summary_records.append(warmup_record)
                        _print_decision_profile_line(episode_idx, warmup_result.request_id, warmup_record)
                        _print_trajectory_bundle(
                            episode_idx,
                            warmup_result.request_id,
                            warmup_result.llm_output,
                            warmup_result.refined_waypoints,
                            current_pose=current_pose,
                        )

                        while not state.dones[0] and control_step < int(args.max_control_steps):
                            applied_result = planner.poll_result()
                            dino_latency_ms = 0.0
                            trajectory_switch_applied = False
                            hover_wait_ms = 0.0
                            dino_predicted_this_step = False

                            if applied_result is not None:
                                applied_result, tcm_extra = _tcm_apply_result(
                                    tcm, state, eval_env, model_wrapper, episode_clock, applied_result
                                )
                                if applied_result is None:
                                    # stale result dropped during target lock — never
                                    # applied, never re-locked.  The NCN interval
                                    # bookkeeping is deliberately left untouched: the
                                    # interval did not end because no buffer was taken.
                                    dropped_record = {
                                        "record_type": "execution",
                                        "exec_step": int(control_step),
                                        "batch_size": 1,
                                        "seq_names": [env_batchs[0]["seq_name"]],
                                        "map_names": [env_batchs[0]["map_name"]],
                                        "chunk_waypoints": int(chunk_waypoints),
                                        "executed_waypoints": 0,
                                        "comm_delay_enabled": bool(enable_comm_delay),
                                        "request_id": None,
                                        "request_submitted_step": None,
                                        "request_observation_timestamp": None,
                                        "request_ready_wall_time": None,
                                        "llm_latency_ms": None,
                                        "traj_latency_ms": None,
                                        "airsim_action_latency_ms": 0.0,
                                        "action_wall_time_ms": 0.0,
                                        "action_sim_time_ms": 0.0,
                                        "obs_latency_ms": 0.0,
                                        "groundingdino_latency_ms": 0.0,
                                        "execution_post_obs_latency_ms": 0.0,
                                        "execution_post_dino_latency_ms": 0.0,
                                        "loop_latency_ms": 0.0,
                                        "decision_latency_ms": None,
                                        "uplink_payload_bits": None,
                                        "uplink_payload_mb": None,
                                        "uplink_bandwidth_mbps": None,
                                        "uplink_latency_ms": None,
                                        "action_age_ms": None,
                                        "state_drift_m": None,
                                        "trajectory_switch_applied": False,
                                        "hover_wait_ms": 0.0,
                                        "llm_output": None,
                                        "refined_waypoints": copy.deepcopy(active_traj),
                                        "current_pose": state.current_sim_pose(),
                                        "first_refined_gap_m": float(_point_delta(active_traj[0], state.current_sim_pose())) if len(active_traj) > 0 else None,
                                        "exec_target_distance_m": None,
                                        "predict_dones": [bool(x) for x in state.predict_dones],
                                        "collisions": [bool(x) for x in state.collisions],
                                        "dones": [bool(x) for x in state.dones],
                                    }
                                    dropped_record.update(tcm_extra)
                                    dropped_record.update(episode_clock.metadata())
                                    _write_jsonl_line(profile_fp, dropped_record)
                                    summary_records.append(dropped_record)
                                else:
                                    edge_latency.update(applied_result.llm_latency_ms, applied_result.traj_latency_ms)
                                    recovered_from_ncn = ncn_actions_in_interval > 0
                                    ncn_actions_in_interval = 0
                                    active_traj = copy.deepcopy(applied_result.refined_waypoints)
                                    active_index = 0
                                    active_result = applied_result
                                    trajectory_switch_applied = True
                                    current_pose = state.current_sim_pose()
                                    decision_record = _build_planner_decision_record(
                                        env_batchs[0],
                                        applied_result,
                                        current_pose,
                                        enable_comm_delay,
                                        chunk_waypoints,
                                        decision_step=int(applied_result.request_id),
                                        clock=episode_clock,
                                        applied_exec_step=control_step,
                                    )
                                    decision_record["predict_dones"] = [bool(x) for x in state.predict_dones]
                                    decision_record["collisions"] = [bool(x) for x in state.collisions]
                                    decision_record["dones"] = [bool(x) for x in state.dones]
                                    decision_record.update(tcm_extra)
                                    if tcm_extra.get("trajectory_mode") == "corrected":
                                        # correction time is non-moving local compute at
                                        # apply — fold it into the approved wait so this
                                        # cell's wait metric stays protocol-comparable
                                        decision_record["hover_wait_ms"] = (
                                            float(decision_record.get("hover_wait_ms", 0.0))
                                            + float(tcm_extra.get("trajcorr_refresh_obs_ms", 0) or 0)
                                            + float(tcm_extra.get("trajcorr_regen_ms", 0) or 0)
                                        )
                                    if recovered_from_ncn:
                                        decision_record["ncn_recovery_reason"] = "valid_edge_buffer_applied"
                                    _write_jsonl_line(profile_fp, decision_record)
                                    summary_records.append(decision_record)
                                    _print_decision_profile_line(episode_idx, applied_result.request_id, decision_record)
                                    _print_trajectory_bundle(
                                        episode_idx,
                                        applied_result.request_id,
                                        applied_result.llm_output,
                                        applied_result.refined_waypoints,
                                        current_pose=current_pose,
                                    )

                            # ── TCM execution-time gate ──────────────────────
                            # Fires while the guidance is still being flown: the
                            # shift that matters is the one accumulated since the
                            # guidance's own uplink, not the one sampled when a
                            # later result happens to land.  A correction
                            # re-observes + regenerates locally and enters the
                            # target lock; the locked buffer is then executed by
                            # the branch below, which also suppresses the NCN
                            # local-action gate for the duration of the lock.
                            if active_result is not None and active_index < len(active_traj):
                                effective, mid_extra = tcm.maybe_correct_mid_execution(
                                    state=state,
                                    eval_env=eval_env,
                                    model_wrapper=model_wrapper,
                                    clock=episode_clock,
                                    active_result=active_result,
                                )
                                if effective is not None:
                                    active_result = effective
                                    active_traj = copy.deepcopy(effective.refined_waypoints)
                                    active_index = 0
                                    trajectory_switch_applied = True
                                    current_pose = state.current_sim_pose()
                                    decision_record = _build_planner_decision_record(
                                        env_batchs[0],
                                        effective,
                                        current_pose,
                                        enable_comm_delay,
                                        chunk_waypoints,
                                        decision_step=int(effective.request_id),
                                        clock=episode_clock,
                                        applied_exec_step=control_step,
                                    )
                                    decision_record["predict_dones"] = [bool(x) for x in state.predict_dones]
                                    decision_record["collisions"] = [bool(x) for x in state.collisions]
                                    decision_record["dones"] = [bool(x) for x in state.dones]
                                    decision_record.update(mid_extra)
                                    decision_record["hover_wait_ms"] = (
                                        float(decision_record.get("hover_wait_ms", 0.0))
                                        + float(mid_extra.get("trajcorr_refresh_obs_ms", 0) or 0)
                                        + float(mid_extra.get("trajcorr_regen_ms", 0) or 0)
                                    )
                                    _write_jsonl_line(profile_fp, decision_record)
                                    summary_records.append(decision_record)
                                    _print_decision_profile_line(episode_idx, effective.request_id, decision_record)
                                    _print_trajectory_bundle(
                                        episode_idx,
                                        effective.request_id,
                                        effective.llm_output,
                                        effective.refined_waypoints,
                                        current_pose=current_pose,
                                    )

                            if active_index < len(active_traj):
                                current_chunk = active_traj[active_index : active_index + 1]
                                chunk_size = len(current_chunk)
                                exec_pose = state.current_sim_pose()
                                exec_target_distance_m = (
                                    float(_point_delta(current_chunk[0], exec_pose)) if len(current_chunk) > 0 else None
                                )
                                action_start = time.perf_counter()
                                eval_env.makeActionsChunk([current_chunk], target_idx=chunk_size)
                                action_end = time.perf_counter()
                                action_times = action_timing(
                                    eval_env,
                                    (action_end - action_start) * 1000.0,
                                    episode_clock.enabled,
                                )
                                episode_clock.advance_action(
                                    action_times["action_sim_time_ms"],
                                    action_times["action_wall_time_ms"],
                                )

                                active_index += chunk_size
                                control_step += 1
                                state.sync_runtime_status_from_sim()
                                check_evaluation(state, model_wrapper, episode_clock, control_step)

                                lock_completion = None
                                if tcm.lock_active:
                                    lock_completion = tcm.target_lock.evaluate(state.current_sim_pose())
                                    if lock_completion is None and active_index >= len(active_traj):
                                        lock_completion = tcm.target_lock.mark_buffer_exhausted()

                                decision_obs_latency_ms = 0.0
                                decision_observation_pose = None
                                should_request_decision = (
                                    not state.dones[0]
                                    and not tcm.lock_active
                                    and (
                                        request_due(control_step, chunk_waypoints)
                                    )
                                )
                                if should_request_decision:
                                    request_counter, _, decision_obs_latency_ms, dino_latency_ms, dino_predicted_this_step = _run_decision_cycle(
                                        state,
                                        eval_env,
                                        model_wrapper,
                                        planner,
                                        request_counter,
                                        control_step,
                                        episode_clock,
                                    )
                                    decision_observation_pose = state.current_sim_pose()

                                action_latency_ms = float(action_times["airsim_action_latency_ms"])
                                record = {
                                    "record_type": "execution",
                                    "exec_step": int(control_step),
                                    "batch_size": 1,
                                    "seq_names": [env_batchs[0]["seq_name"]],
                                    "map_names": [env_batchs[0]["map_name"]],
                                    "chunk_waypoints": int(chunk_waypoints),
                                    "executed_waypoints": int(chunk_size),
                                    "comm_delay_enabled": bool(enable_comm_delay),
                                    "request_id": None,
                                    "request_submitted_step": None,
                                    "request_observation_timestamp": None,
                                    "request_ready_wall_time": None,
                                    "llm_latency_ms": None,
                                    "traj_latency_ms": None,
                                    "airsim_action_latency_ms": action_latency_ms,
                                    "action_wall_time_ms": float(action_times["action_wall_time_ms"]),
                                    "action_sim_time_ms": float(action_times["action_sim_time_ms"]),
                                    "obs_latency_ms": 0.0,
                                    "groundingdino_latency_ms": 0.0,
                                    "execution_post_obs_latency_ms": decision_obs_latency_ms,
                                    "execution_post_dino_latency_ms": float(dino_latency_ms),
                                    "loop_latency_ms": action_latency_ms,
                                    "decision_latency_ms": None,
                                    "uplink_payload_bits": None,
                                    "uplink_payload_mb": None,
                                    "uplink_bandwidth_mbps": None,
                                    "uplink_latency_ms": None,
                                    "action_age_ms": None,
                                    "state_drift_m": None,
                                    "trajectory_switch_applied": bool(trajectory_switch_applied),
                                    "hover_wait_ms": float(hover_wait_ms),
                                    "llm_output": None,
                                    "refined_waypoints": copy.deepcopy(active_traj),
                                    "current_pose": state.current_sim_pose(),
                                    "first_refined_gap_m": float(_point_delta(active_traj[0], state.current_sim_pose())) if len(active_traj) > 0 else None,
                                    "exec_target_distance_m": exec_target_distance_m,
                                    "predict_dones": [bool(x) for x in state.predict_dones],
                                    "collisions": [bool(x) for x in state.collisions],
                                    "dones": [bool(x) for x in state.dones],
                                }
                                if tcm.lock_active:
                                    record["target_lock_active"] = True
                                if lock_completion is not None:
                                    record["target_lock_completion_reason"] = lock_completion
                                record.update(episode_clock.metadata())
                                _write_jsonl_line(profile_fp, record)
                                summary_records.append(record)
                                _print_exec_profile_line(
                                    episode_idx,
                                    control_step,
                                    record,
                                    current_chunk=current_chunk,
                                    current_pose=exec_pose,
                                )
                                if lock_completion is not None:
                                    tcm.stats.lock_completions[lock_completion] += 1
                                    active_traj = []
                                    active_index = 0
                                    active_result = None
                                    if not state.dones[0]:
                                        request_counter, _, _, _, _ = _run_decision_cycle(state, eval_env, model_wrapper, planner, request_counter, control_step, episode_clock, reason="module_refresh")
                                if state.dones[0] and dino_predicted_this_step and not state.collisions[0]:
                                    stop_pose = state.current_sim_pose()
                                    stop_record = {
                                        "record_type": "decision_stop",
                                        "decision_step": int(control_step),
                                        "batch_size": 1,
                                        "seq_names": [env_batchs[0]["seq_name"]],
                                        "map_names": [env_batchs[0]["map_name"]],
                                        "chunk_waypoints": int(chunk_waypoints),
                                        "executed_waypoints": int(chunk_size),
                                        "comm_delay_enabled": bool(enable_comm_delay),
                                        "request_id": None,
                                        "submitted_after_exec_step": int(control_step),
                                        "obs_latency_ms": decision_obs_latency_ms,
                                        "groundingdino_latency_ms": float(dino_latency_ms),
                                        "uplink_latency_ms": None,
                                        "llm_latency_ms": None,
                                        "traj_latency_ms": None,
                                        "decision_latency_ms": float(decision_obs_latency_ms + dino_latency_ms),
                                        "airsim_action_latency_ms": 0.0,
                                        "loop_latency_ms": float(decision_obs_latency_ms + dino_latency_ms),
                                        "action_age_ms": None,
                                        "state_drift_m": None,
                                        "trajectory_switch_applied": False,
                                        "hover_wait_ms": 0.0,
                                        "current_pose": copy.deepcopy(stop_pose),
                                        "predict_done": True,
                                        "stop_delay_ms": float(dino_latency_ms),
                                        "stop_drift_m": float(_point_delta(decision_observation_pose, stop_pose)),
                                        "predict_dones": [bool(x) for x in state.predict_dones],
                                        "collisions": [bool(x) for x in state.collisions],
                                        "dones": [bool(x) for x in state.dones],
                                    }
                                    _write_jsonl_line(profile_fp, stop_record)
                                    summary_records.append(stop_record)
                                    _print_decision_profile_line(episode_idx, control_step, stop_record)
                            else:
                                if tcm.lock_active:
                                    # defensive: lock with an empty buffer ends the lock
                                    completion = tcm.target_lock.mark_buffer_exhausted()
                                    tcm.stats.lock_completions[completion] += 1
                                    active_traj = []
                                    active_index = 0
                                    active_result = None
                                    if not state.dones[0]:
                                        request_counter, _, _, _, _ = _run_decision_cycle(state, eval_env, model_wrapper, planner, request_counter, control_step, episode_clock, reason="module_refresh")
                                    continue

                                if state.dones[0]:
                                    state.maybe_finalize()
                                    break


                                predicted_remaining_ms = planner.predicted_remaining_ms(edge_latency.compute_ms)
                                if (
                                    ncn_runtime is not None
                                    and not tcm.lock_active
                                    and predicted_remaining_ms is not None
                                    and predicted_remaining_ms > float(args.ncn_response_safety_margin_ms)
                                    and ncn_actions_in_interval < int(args.ncn_max_consecutive_actions)
                                ):
                                    record = _run_ncn_action(
                                        state, eval_env, ncn_runtime,
                                        env_batchs[0]["instruction"], episode_clock,
                                        control_step + 1, predicted_remaining_ms, 0.0,
                                        "buffer_insufficient_for_predicted_edge_response",
                                    )
                                    control_step += 1
                                    if request_due(control_step, chunk_waypoints) and not state.dones[0]:
                                        request_counter, _, _, _, _ = _run_decision_cycle(state, eval_env, model_wrapper, planner, request_counter, control_step, episode_clock)
                                    ncn_actions_in_interval += 1
                                    _write_jsonl_line(profile_fp, record)
                                    summary_records.append(record)
                                    _print_exec_profile_line(
                                        episode_idx, control_step, record,
                                        current_chunk=[[record["aerodpo_action"][0]["fwd"], record["aerodpo_action"][0]["down"], record["aerodpo_action"][0]["yaw"]]],
                                        current_pose=state.current_sim_pose(),
                                    )
                                    if state.dones[0]:
                                        state.maybe_finalize()
                                    continue

                                hover_wait_start = time.perf_counter()
                                hover_logical_start = episode_clock.now_ms
                                applied_result = planner.wait_result()
                                hover_wait_ms = (
                                    episode_clock.now_ms - hover_logical_start
                                    if episode_clock.enabled
                                    else (time.perf_counter() - hover_wait_start) * 1000.0
                                )
                                applied_result, tcm_extra = _tcm_apply_result(
                                    tcm, state, eval_env, model_wrapper, episode_clock, applied_result
                                )
                                if applied_result is None:
                                    # stale-drop in the hover branch: loop again (next poll applies fresh)
                                    continue
                                active_traj = copy.deepcopy(applied_result.refined_waypoints)
                                active_index = 0
                                active_result = applied_result
                                current_pose = state.current_sim_pose()
                                record = _build_planner_decision_record(
                                    env_batchs[0],
                                    applied_result,
                                    current_pose,
                                    enable_comm_delay,
                                    chunk_waypoints,
                                    decision_step=int(applied_result.request_id),
                                    clock=episode_clock,
                                    applied_exec_step=control_step,
                                )
                                record["hover_wait_ms"] = float(hover_wait_ms)
                                record["loop_latency_ms"] = float(hover_wait_ms)
                                record["predict_dones"] = [bool(x) for x in state.predict_dones]
                                record["collisions"] = [bool(x) for x in state.collisions]
                                record["dones"] = [bool(x) for x in state.dones]
                                record.update(tcm_extra)
                                if tcm_extra.get("trajectory_mode") == "corrected":
                                    record["hover_wait_ms"] = (
                                        float(record.get("hover_wait_ms", 0.0))
                                        + float(tcm_extra.get("trajcorr_refresh_obs_ms", 0) or 0)
                                        + float(tcm_extra.get("trajcorr_regen_ms", 0) or 0)
                                    )
                                _write_jsonl_line(profile_fp, record)
                                summary_records.append(record)
                                _print_decision_profile_line(episode_idx, applied_result.request_id, record)
                                _print_trajectory_bundle(
                                    episode_idx,
                                    applied_result.request_id,
                                    applied_result.llm_output,
                                    applied_result.refined_waypoints,
                                    current_pose=current_pose,
                                )

                            if state.dones[0]:
                                state.maybe_finalize()
                                break

                        if not state.skip_saved:
                            state.maybe_finalize(force=True)
                        episode_record = {
                            "record_type": "episode_end",
                            "initial_obs_latency_ms": float(eval_env.initial_obs_latency_ms),
                            "initial_uplink_start_logical_ms": float(warmup_result.submitted_logical_ms),
                            "initial_uplink_latency_ms": float(warmup_result.uplink_latency_ms),
                            **state.cr_state.metadata(),
                            "seq_names": [env_batchs[0]["seq_name"]],
                            "map_names": [env_batchs[0]["map_name"]],
                            "control_steps": int(control_step),
                            "episode_latency_ms": float(episode_clock.now_ms),
                            "success": bool(state.success),
                            "oracle_success": bool(state.oracle_success),
                            "collision": bool(state.collisions[0]),
                            "final_ne_m": float(state.distance_to_ends[-1]) if state.distance_to_ends else None,
                        }
                        episode_record.update(tcm.stats.as_dict())
                        episode_record.update(episode_clock.metadata())
                        _write_jsonl_line(profile_fp, episode_record)
                        summary_records.append(episode_record)
                        episode_ok = True
                        break
                    except Exception as e:
                        logger.error(f"Episode failed: {e}, retry {retry_i + 1}/3")
                        if retry_i < 2:
                            logger.error("Restarting scene...")
                            eval_env._changeEnv(need_change=True)
                    finally:
                        if planner is not None:
                            planner.close()

                if not episode_ok:
                    raise RuntimeError("episode failed after 3 retries")
                episode_idx += 1

        try:
            pbar.close()
        except Exception:
            pass
    execution_records = [item for item in summary_records if item.get("record_type") == "execution"]
    ncn_records = [item for item in execution_records if item.get("control_source") == "ncn"]
    decision_records = [item for item in summary_records if item.get("record_type") in ("decision", "decision_stop")]
    episode_records = [item for item in summary_records if item.get("record_type") == "episode_end"]
    trigger_episodes = sum(1 for item in episode_records if item.get("tcm_corrected_applies", 0) > 0)
    ncn_episode_ids = {item["seq_names"][0] for item in ncn_records if item.get("seq_names")}
    summary = {
        "execution_airsim_action_latency_ms": _metric_summary([item.get("airsim_action_latency_ms") for item in execution_records]),
        "execution_loop_latency_ms": _metric_summary([item.get("loop_latency_ms") for item in execution_records]),
        "execution_target_distance_m": _metric_summary([item.get("exec_target_distance_m") for item in execution_records]),
        "decision_obs_latency_ms": _metric_summary([item.get("obs_latency_ms") for item in decision_records]),
        "decision_groundingdino_latency_ms": _metric_summary([item.get("groundingdino_latency_ms") for item in decision_records]),
        "decision_uplink_latency_ms": _metric_summary([item.get("uplink_latency_ms") for item in decision_records]),
        "decision_llm_latency_ms": _metric_summary([item.get("llm_latency_ms") for item in decision_records]),
        "decision_traj_latency_ms": _metric_summary([item.get("traj_latency_ms") for item in decision_records]),
        "decision_total_latency_ms": _metric_summary([item.get("decision_latency_ms") for item in decision_records]),
        "decision_uplink_payload_bits": _metric_summary([item.get("uplink_payload_bits") for item in decision_records]),
        "decision_uplink_payload_mb": _metric_summary([item.get("uplink_payload_mb") for item in decision_records]),
        "decision_uplink_bandwidth_mbps": _metric_summary([item.get("uplink_bandwidth_mbps") for item in decision_records]),
        "action_age_ms": _metric_summary([item.get("action_age_ms") for item in decision_records]),
        "state_drift_m": _metric_summary([item.get("state_drift_m") for item in decision_records]),
        "stop_delay_ms": _metric_summary([item.get("stop_delay_ms") for item in decision_records]),
        "stop_drift_m": _metric_summary([item.get("stop_drift_m") for item in decision_records]),
        "hover_wait_ms": _metric_summary([item.get("hover_wait_ms") for item in decision_records]),
        "episode_latency_ms": _metric_summary([item.get("episode_latency_ms") for item in episode_records]),
        "num_execution_records": len(execution_records),
        "num_decision_records": len(decision_records),
        "num_records": len(summary_records),
        "ncn_enabled": bool(ncn_runtime is not None),
        "ncn_action_count": len(ncn_records),
        "ncn_activation_rate": float(len(ncn_episode_ids) / len(episode_records)) if episode_records else 0.0,
        "ncn_image_text_to_action_latency_ms": _metric_summary([item.get("image_text_to_action_latency_ms") for item in ncn_records]),
        "ncn_trigger_reason_counts": {
            reason: sum(1 for item in ncn_records if item.get("ncn_trigger_reason") == reason)
            for reason in sorted({item.get("ncn_trigger_reason") for item in ncn_records if item.get("ncn_trigger_reason")})
        },
        "trajcorr_enabled": bool(trajcorr_enabled),
        "trajcorr_trigger_episodes": int(trigger_episodes),
        "trajcorr_trigger_rate": float(trigger_episodes / len(episode_records)) if episode_records else 0.0,
        "trajcorr_corrected_applies": int(sum(item.get("tcm_corrected_applies", 0) for item in episode_records)),
        "trajcorr_mid_exec_corrections": int(sum(item.get("tcm_mid_exec_corrections", 0) for item in episode_records)),
        "trajcorr_stale_dropped": int(sum(item.get("tcm_stale_dropped", 0) for item in episode_records)),
        "trajcorr_lock_completions": {
            reason: int(sum(item.get("tcm_lock_completions", {}).get(reason, 0) for item in episode_records))
            for reason in ("goal_reached", "goal_passed", "buffer_exhausted")
        },
        "trajcorr_correction_fallbacks": {
            reason: int(sum(item.get("tcm_correction_fallbacks", {}).get(reason, 0) for item in episode_records))
            for reason in sorted({
                reason
                for item in episode_records
                for reason in (item.get("tcm_correction_fallbacks") or {})
            })
        },
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    if ncn_runtime is not None:
        ncn_runtime.close()


if __name__ == "__main__":
    _set_console_log_message_only()
    if not _as_bool(args.enable_ncn):
        raise ValueError("continue_ncn.py requires --enable_ncn True")
    configure_fast_eval_output(args, "continuous_ncn")
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path

    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)
    profile_log_dir = os.path.join(eval_save_path, "profile_logs")
    os.makedirs(profile_log_dir, exist_ok=True)

    enable_comm_delay = _as_bool(args.enable_comm_delay)
    trace_path = Path(args.comm_trace_csv_path) if args.comm_trace_csv_path else Path(default_trace_path())
    if not trace_path.exists():
        raise FileNotFoundError(f"bandwidth trace not found: {trace_path}")
    chunk_waypoints = max(1, int(args.chunk_waypoints))
    trajcorr_enabled = str(getattr(args, "trajcorr_mode", "on")).strip().lower() == "on"
    delta_cor_m = float(getattr(args, "trajcorr_state_shift_threshold_m", 2.5))
    bandwidth_trace = BandwidthTrace(trace_path, cycle=True)
    logger.info(
        f"Loaded bandwidth trace: {trace_path} ({bandwidth_trace.sample_count} samples), "
        f"enable_comm_delay={enable_comm_delay}, chunk_waypoints={chunk_waypoints}, "
        f"trajcorr_mode={args.trajcorr_mode}, trajcorr_state_shift_threshold_m={delta_cor_m}"
    )

    setup()

    assert CheckPort(), "error port"

    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False

    model_wrapper = MatchContinuousTravelModelWrapper(model_args=model_args, data_args=data_args)
    model_wrapper.dino_moinitor = DinoMonitor.get_instance()

    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)

    print("Assist setting: always_help --", args.always_help, "    use_gt --", args.use_gt)

    profile_log_path = os.path.join(profile_log_dir, f"pro_con_w{chunk_waypoints}_{args.make_dir_time}.jsonl")
    summary_path = os.path.join(profile_log_dir, f"pro_con_w{chunk_waypoints}_{args.make_dir_time}_summary.json")

    eval(
        model_wrapper=model_wrapper,
        assist=assist,
        eval_env=eval_env,
        eval_save_dir=eval_save_path,
        profile_log_path=profile_log_path,
        summary_path=summary_path,
        bandwidth_trace=bandwidth_trace,
        enable_comm_delay=enable_comm_delay,
        chunk_waypoints=chunk_waypoints,
        trajcorr_enabled=trajcorr_enabled,
        delta_cor_m=delta_cor_m,
    )

    eval_env.delete_VectorEnvUtil()
