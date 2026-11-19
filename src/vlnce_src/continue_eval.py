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
from src.vlnce_src.continuous_runtime import FixedStepEdgePlanner as LatestOnlyEdgePlanner, run_decision_cycle as _run_decision_cycle, request_due

from src.vlnce_src.collision_state import EpisodeCollisionState
from assist import Assist
from src.common.param import args, data_args, model_args
from src.model_wrapper.profile_travel_llm import ProfileTravelModelWrapper
from src.vlnce_src.closeloop_util import (
    BatchIterator,
    CheckPort,
    initialize_env_eval,
    is_dist_avail_and_initialized,
    finalize_eval_episode,
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
from src.vlnce_src.eval_contract import check_collision_without_tiny_diff
from env_uav import AirVLNENV
from utils.logger import logger
from utils.utils import *


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
    coarse_local: List[float]
    coarse_goal_world: List[float]
    llm_output: List[List[float]]
    refined_waypoints: List[List[float]]
    submitted_logical_ms: Optional[float] = None
    edge_arrival_logical_ms: Optional[float] = None
    ready_logical_ms: Optional[float] = None


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


class ContinuousEpisodeState:
    """Per-episode state for the Continuous evaluation paradigm.

    ============
    TERMINATION — see ``rule_based_eval.py`` ``ContinuousEpisodeState`` for the
    authoritative contract.  This class shares the same three termination
    sources (collision / DINO / NE-trend) plus safety caps.  DO NOT modify
    termination logic here without updating ALL paradigms.
    ============
    """

    def __init__(self, env_batch, eval_env: AirVLNENV, assist: Assist, ignore_tiny_diff: bool = False):
        self.eval_env = eval_env
        self.assist = assist
        self.ignore_tiny_diff = ignore_tiny_diff
        self.env_batch = env_batch
        self.target_position = env_batch["object_position"]
        self.traj = env_batch["trajectory"]
        self.ori_data_dir = env_batch["trajectory_dir"]
        self.object_info = _load_object_description().get(env_batch["object"]["asset_name"].replace("AA", ""))

        self.episode: List[Dict[str, Any]] = []
        self.dones = [False]
        self.collisions = [False]
        self.cr_state = EpisodeCollisionState()
        self.predict_dones = [False]
        self.success = False
        self.oracle_success = False
        self.early_end = False
        self.skip_saved = False
        self.distance_to_ends: List[float] = []
        self.request_ne_history: List[float] = []
        self.last_observation_timestamp: Optional[int] = None

        outputs = self.eval_env.reset()
        self.process_env_output(outputs)
        self.request_ne_history = [self._calculate_distance_from_position(self.current_sim_pose())]
        self._evaluation_checked_step = 0

    def _check_collision_for_continuous(self, observations, dones, collisions):
        if int(getattr(self.eval_env.sim_states[0], "step", 0)) == 0:
            return collisions, dones
        if not self.ignore_tiny_diff:
            return self.assist.check_collision_by_depth([self.episode], observations, collisions, dones)
        return check_collision_without_tiny_diff([self.episode], observations, collisions, dones)

    def _append_unique_observations(self, observations: List[Dict[str, Any]]) -> None:
        for observation in observations:
            ts = observation["sensors"]["state"]["timestamp"]
            if self.last_observation_timestamp is not None and ts <= self.last_observation_timestamp:
                continue
            self.episode.append(observation)
            self.last_observation_timestamp = ts
            self.distance_to_ends.append(self._calculate_distance(observation))

    def _calculate_distance(self, observation: Dict[str, Any]) -> float:
        return float(
            np.linalg.norm(
                np.asarray(observation["sensors"]["state"]["position"], dtype=np.float64)
                - np.asarray(self.target_position, dtype=np.float64)
            )
        )

    REQUEST_REGRESSION_COUNT = 10

    def record_request_ne(self):
        """Compatibility hook; NE is checked only by the five-step evaluator."""
        return None

    def _calculate_distance_from_position(self, position) -> float:
        return float(
            np.linalg.norm(
                np.asarray(position, dtype=np.float64)
                - np.asarray(self.target_position, dtype=np.float64)
            )
        )

    def current_sim_pose(self) -> List[float]:
        return copy.deepcopy(self.eval_env.sim_states[0].pose[0:3])

    def sync_runtime_status_from_sim(self) -> None:
        sim_state = self.eval_env.sim_states[0]
        if self.cr_state.observe(sim_state.trajectory[-1:]):
            self.collisions[0] = True
            self.dones[0] = True
        if bool(getattr(sim_state, "is_end", False)):
            self.dones[0] = True
        if bool(getattr(sim_state, "oracle_success", False)):
            self.oracle_success = True
        self.distance_to_ends.append(self._calculate_distance_from_position(self.current_sim_pose()))

    def process_env_output(self, outputs) -> None:
        observations, dones, collisions, oracle_success = [list(x) for x in zip(*outputs)]
        collision_flags, done_flags = self._check_collision_for_continuous(observations, dones, collisions)
        self.dones = done_flags
        self.collisions = [self.cr_state.observe(observations[0])]
        if self.collisions[0]:
            self.dones[0] = True
        self.oracle_success = bool(oracle_success[0])
        self._append_unique_observations(observations[0])

    def build_snapshot(
        self,
        request_id: int,
        submitted_step: int,
        obs_latency_ms: float = 0.0,
        groundingdino_latency_ms: float = 0.0,
    ) -> Snapshot:
        latest = self.episode[-1]
        return Snapshot(
            request_id=request_id,
            submitted_step=submitted_step,
            episode=copy.deepcopy(self.episode),
            target_position=copy.deepcopy(self.target_position),
            object_info=self.object_info,
            assist_notice=self.assist.get_assist_notice(
                [self.episode],
                [self.traj],
                [self.object_info],
                [self.target_position],
            )[0],
            observation_timestamp=int(latest["sensors"]["state"]["timestamp"]),
            observation_pose=copy.deepcopy(latest["sensors"]["state"]["position"][0:3]),
            submitted_perf_time=time.perf_counter(),
            submitted_wall_time=time.time(),
            obs_latency_ms=float(obs_latency_ms),
            groundingdino_latency_ms=float(groundingdino_latency_ms),
        )

    def run_dino_and_update_metric(self, model_wrapper: ProfileTravelModelWrapper) -> float:
        dino_start = time.perf_counter()
        self.predict_dones = model_wrapper.predict_done([self.episode], [self.object_info])
        dino_latency_ms = (time.perf_counter() - dino_start) * 1000.0
        if not self.dones[0] and self.predict_dones[0]:
            current_distance = self.distance_to_ends[-1]
            if current_distance <= 20:
                self.success = True
                self.dones[0] = True
            elif current_distance > 20:
                self.early_end = True
        return dino_latency_ms

    def maybe_finalize(self, force=False) -> bool:
        self.skip_saved = finalize_eval_episode(
            self.episode, self.ori_data_dir, self.success, self.oracle_success,
            self.dones[0], self.skip_saved, force,
        )
        return self.skip_saved






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
):
    from src.vlnce_src.continuous_runtime import evaluate
    return evaluate(model_wrapper, assist, eval_env, eval_save_dir, profile_log_path, summary_path, bandwidth_trace, enable_comm_delay, chunk_waypoints, mode="continuous")


if __name__ == "__main__":
    _set_console_log_message_only()
    configure_fast_eval_output(args, "continuous")
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
    bandwidth_trace = BandwidthTrace(trace_path, cycle=True)
    logger.info(
        f"Loaded bandwidth trace: {trace_path} ({bandwidth_trace.sample_count} samples), "
        f"enable_comm_delay={enable_comm_delay}, chunk_waypoints={chunk_waypoints}"
    )

    setup()

    assert CheckPort(), "error port"

    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False

    model_wrapper = ProfileTravelModelWrapper(model_args=model_args, data_args=data_args)
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
    )

    eval_env.delete_VectorEnvUtil()
