"""Logical-arrival ordered edge execution; scheduling policy belongs to callers."""
import copy, json, math, time
from threading import Condition, Lock, Thread
from types import SimpleNamespace as NS
import numpy as np
import torch
from src.common.param import args
from src.vlnce_src.comm_delay import calculate_latency_ms, estimate_uplink_payload_bits_from_episodes
from src.vlnce_src.fast_eval_time import FastEvalClock, action_timing
from src.vlnce_src.closeloop_util import target_distance_increasing_for_10frames

def request_due(step, interval):
    return step > 0 and step % max(1, int(interval)) == 0

def capture(state, clock):
    """Capture with the existing depth check, including the five-step evaluator."""
    return _capture(state, clock, state.process_env_output)

def capture_observation(state, clock):
    """Capture for asynchronous requests; evaluation runs separately every five steps."""
    return _capture(state, clock, state.record_env_output)

def _capture(state, clock, process_output):
    start = time.perf_counter()
    process_output(state.eval_env.get_obs())
    state._last_capture_step = int(state.eval_env.sim_states[0].step)
    ms = (time.perf_counter() - start) * 1000
    clock.advance_blocking(ms)
    return ms

def check_evaluation(state, model, clock, step, write=lambda r: None, *, account_ne=False):
    if step == 0 or step % 5 or step == getattr(state, '_evaluation_checked_step', 0):
        return None
    state._evaluation_checked_step = step
    obs = capture(state, clock)
    ne_start = time.perf_counter() if account_ne else None
    ne = state._calculate_distance_from_position(state.current_sim_pose())
    state.request_ne_history.append(ne)
    if target_distance_increasing_for_10frames(state.request_ne_history):
        state.dones[0] = True
    ne_ms = (time.perf_counter() - ne_start) * 1000.0 if account_ne else 0.0
    if account_ne:
        clock.advance_blocking(ne_ms)
    dino = 0.0
    if not state.dones[0]:
        dino = state.run_dino_and_update_metric(model)
        clock.advance_blocking(dino)
    evaluation_fields = {'ne_check_latency_ms': ne_ms} if account_ne else {}
    write(dict(record_type='evaluation_check', control_step=step, ne_m=ne, obs_latency_ms=obs, groundingdino_latency_ms=dino, **evaluation_fields, **clock.metadata()))
    return obs

def run_decision_cycle(state, eval_env, model, planner, counter, step, clock, reason='periodic'):
    obs = capture(state, clock)
    if not state.dones[0]:
        snap = state.build_snapshot(counter + 1, step, obs, 0.0)
        planner.submit(snap, reason)
        counter = snap.request_id
    return (counter, None, obs, 0.0, False)

class AsyncEdgePlanner:

    def __init__(self, model_wrapper, bandwidth_trace, enable_comm_delay, clock):
        self.model_wrapper = model_wrapper
        self.trace = bandwidth_trace
        self.enable = enable_comm_delay
        self.clock = clock
        self.timeout_ms = float(args.continuous_request_timeout_ms)
        self.jobs = []
        self.next_id = 0
        self.compute_end = 0.0
        self.state = None
        self.capture_observation = capture
        self.record_event = lambda r: None

    def bind_state(self, state, record_event):
        self.state = state
        self.record_event = record_event

    def submit(self, snapshot, reason='periodic'):
        start = self.clock.now_ms
        _, bits, mb = estimate_uplink_payload_bits_from_episodes([snapshot.episode])
        bw = self.trace.bandwidth_at_ms(start)
        uplink = calculate_latency_ms(bits, bw) if self.enable else 0.0
        snapshot.request_id = self.next_id
        self.next_id += 1
        snapshot.submitted_logical_ms = start
        snapshot.planner_started_logical_ms = start
        self.jobs.append(NS(snapshot=snapshot, start=start, arrival=start + uplink, deadline=start + self.timeout_ms, bits=bits, mb=mb, bw=bw, uplink=uplink, result=None, started=False))
        self.record_event(dict(record_type='request', request_id=snapshot.request_id, request_reason=reason, submitted_after_exec_step=snapshot.submitted_step, uplink_start_logical_ms=start, edge_arrival_logical_ms=start + uplink, obs_latency_ms=snapshot.obs_latency_ms, uplink_latency_ms=uplink, uplink_bandwidth_mbps=bw / 1000000.0, **self.clock.metadata()))
        return True

    def has_inflight(self):
        return bool(self.jobs)

    def has_pending_or_running(self):
        return bool(self.jobs)

    def predicted_remaining_ms(self, estimated_compute_ms):
        if not self.jobs:
            return None
        return max(0, min((j.result.ready_logical_ms if j.result else max(j.arrival, self.compute_end) + estimated_compute_ms for j in self.jobs)) - self.clock.now_ms)

    def _pump(self):
        while True:
            pending = [j for j in self.jobs if not j.started and math.isfinite(j.arrival) and (j.arrival <= j.deadline)]
            if not pending:
                return
            j = min(pending, key=lambda j: (j.arrival, j.snapshot.request_id))
            begin = max(j.arrival, self.compute_end)
            if begin > self.clock.now_ms or begin > j.deadline:
                return
            j.started = True
            s = j.snapshot
            inputs, rot = self.model_wrapper.prepare_inputs([s.episode], [s.target_position], [s.assist_notice])
            with torch.inference_mode():
                refined, p = self.model_wrapper.run_profiled(inputs=inputs, episodes=[s.episode], rot_to_targets=rot)
            ready = begin + p['llm_latency_ms'] + p['traj_latency_ms']
            self.compute_end = ready
            j.result = NS(**vars(s), ready_wall_time=time.time(), llm_latency_ms=p['llm_latency_ms'], traj_latency_ms=p['traj_latency_ms'], uplink_payload_bits=j.bits, uplink_payload_mb=j.mb, uplink_bandwidth_mbps=j.bw / 1000000.0, uplink_latency_ms=j.uplink, llm_output=copy.deepcopy(p['llm_output']), coarse_local=copy.deepcopy(p['coarse_local'][0]), coarse_goal_world=copy.deepcopy(p['coarse_goal_world'][0]), refined_waypoints=np.asarray(refined[0]).tolist(), edge_arrival_logical_ms=j.arrival, ready_logical_ms=ready)
            self.record_event(dict(record_type='edge_compute', request_id=s.request_id, edge_arrival_logical_ms=j.arrival, compute_start_logical_ms=begin, result_ready_logical_ms=ready, **self.clock.metadata()))

    def poll_result(self):
        self._pump()
        ready = [j for j in self.jobs if j.result and j.result.ready_logical_ms <= min(self.clock.now_ms, j.deadline)]
        if not ready:
            return None
        j = min(ready, key=lambda j: (j.result.ready_logical_ms, j.snapshot.request_id))
        self.jobs.remove(j)
        return j.result

    def advance(self, ms):
        target = self.clock.now_ms + ms
        if self.clock.enabled:
            self.clock.advance_to(target)
        else:
            time.sleep(max(0, ms) / 1000)
        self._pump()

    def retry_expired(self):
        expired = [j for j in self.jobs if j.deadline <= self.clock.now_ms and (j.result is None or j.result.ready_logical_ms > j.deadline)]
        for j in expired:
            self.jobs.remove(j)
            self.record_event(dict(record_type='request_timeout', request_id=j.snapshot.request_id, submitted_after_exec_step=self.control_step(), **self.clock.metadata()))
            obs = self.capture_observation(self.state, self.clock)
            if not self.state.dones[0]:
                self.submit(self.state.build_snapshot(self.next_id, self.control_step(), obs, 0.0), 'timeout_retry')
        return bool(expired)

    def wait_result(self):
        while self.jobs:
            r = self.poll_result()
            if r:
                return r
            expired = self.retry_expired()
            if self.state and self.state.dones[0]:
                return None
            if expired:
                continue
            events = [j.deadline for j in self.jobs]
            for j in self.jobs:
                events.append(j.result.ready_logical_ms if j.result else max(j.arrival, self.compute_end))
            target = min((t for t in events if t > self.clock.now_ms))
            self.advance(target - self.clock.now_ms)
        raise RuntimeError('No request to wait for')

    def close(self):
        self.jobs.clear()
FixedStepEdgePlanner = AsyncEdgePlanner


class NavigationEdgePlanner(AsyncEdgePlanner):
    """One edge server for Continuous-only and Random, with two clock modes."""

    def __init__(self, model_wrapper, bandwidth_trace, enable_comm_delay, clock):
        super().__init__(model_wrapper, bandwidth_trace, enable_comm_delay, clock)
        self.capture_observation = capture_observation
        self._changed = Condition()
        self._closed = False
        self._error = None
        self._latest_applied_id = -1
        self._worker = None
        if not clock.enabled:
            self._worker = Thread(target=self._run_wall_worker, daemon=True)
            self._worker.start()

    def submit(self, snapshot, reason='periodic'):
        with self._changed:
            result = super().submit(snapshot, reason)
            self._changed.notify_all()
            return result

    def _next_job(self, edge_free_ms):
        eligible = [
            job for job in self.jobs
            if not job.started and math.isfinite(job.arrival)
            and max(job.arrival, edge_free_ms) <= job.deadline
        ]
        return min(eligible, key=lambda job: (job.arrival, job.snapshot.request_id), default=None)

    def _process_job(self, job, begin):
        snapshot = job.snapshot
        start = time.perf_counter()
        with torch.inference_mode():
            inputs, rotations = self.model_wrapper.prepare_inputs(
                [snapshot.episode], [snapshot.target_position], [snapshot.assist_notice],
            )
            waypoints, profile = self.model_wrapper.run_profiled(
                inputs=inputs, episodes=[snapshot.episode], rot_to_targets=rotations,
            )
            refined_waypoints = np.asarray(waypoints[0]).tolist()
        service_ms = (time.perf_counter() - start) * 1000.0
        ready = begin + service_ms if self.clock.enabled else self.clock.now_ms
        return NS(
            **vars(snapshot),
            ready_wall_time=time.time(),
            llm_latency_ms=profile['llm_latency_ms'],
            traj_latency_ms=profile['traj_latency_ms'],
            edge_service_ms=service_ms,
            uplink_payload_bits=job.bits,
            uplink_payload_mb=job.mb,
            uplink_bandwidth_mbps=job.bw / 1000000.0,
            uplink_latency_ms=job.uplink,
            refined_waypoints=refined_waypoints,
            edge_arrival_logical_ms=job.arrival,
            ready_logical_ms=ready,
        )

    def _publish(self, job, begin, result):
        self.compute_end = result.ready_logical_ms
        job.result = result
        self.record_event(dict(
            record_type='edge_compute',
            request_id=job.snapshot.request_id,
            edge_arrival_logical_ms=job.arrival,
            compute_start_logical_ms=begin,
            edge_service_ms=result.edge_service_ms,
            llm_latency_ms=result.llm_latency_ms,
            traj_latency_ms=result.traj_latency_ms,
            result_ready_logical_ms=result.ready_logical_ms,
            **self.clock.metadata(),
        ))

    def _pump(self):
        if not self.clock.enabled:
            return
        while True:
            job = self._next_job(self.compute_end)
            if job is None:
                return
            begin = max(job.arrival, self.compute_end)
            if begin > self.clock.now_ms:
                return
            job.started = True
            self._publish(job, begin, self._process_job(job, begin))

    def _run_wall_worker(self):
        try:
            while True:
                with self._changed:
                    while not self._closed:
                        now = self.clock.now_ms
                        job = self._next_job(now)
                        if job is None:
                            self._changed.wait()
                        elif job.arrival > now:
                            self._changed.wait((job.arrival - now) / 1000.0)
                        else:
                            job.started = True
                            break
                    else:
                        return
                result = self._process_job(job, now)
                with self._changed:
                    if self._closed:
                        return
                    result.ready_logical_ms = self.clock.now_ms
                    self._publish(job, now, result)
                    self._changed.notify_all()
        except Exception as error:
            with self._changed:
                self._error = error
                self._changed.notify_all()

    def _take_ready(self):
        if self._error is not None:
            raise self._error
        now = self.clock.now_ms
        ready = [
            job for job in self.jobs if job.result is not None
            and job.result.ready_logical_ms <= min(now, job.deadline)
        ]
        if not ready:
            return None
        newest = max(ready, key=lambda job: job.snapshot.request_id)
        use_newest = newest.snapshot.request_id > self._latest_applied_id
        for job in ready:
            self.jobs.remove(job)
            if job is not newest or not use_newest:
                self.record_event(dict(
                    record_type='stale_result',
                    request_id=job.snapshot.request_id,
                    **self.clock.metadata(),
                ))
        if use_newest:
            self._latest_applied_id = newest.snapshot.request_id
            return newest.result
        return None

    def poll_result(self):
        self._pump()
        with self._changed:
            return self._take_ready()

    def retry_expired(self):
        with self._changed:
            now = self.clock.now_ms
            expired = [
                job for job in self.jobs if job.deadline <= now
                and (job.result is None or job.result.ready_logical_ms > job.deadline)
            ]
            for job in expired:
                self.jobs.remove(job)
        for job in expired:
            self.record_event(dict(
                record_type='request_timeout', request_id=job.snapshot.request_id,
                submitted_after_exec_step=self.control_step(), **self.clock.metadata(),
            ))
            if job.snapshot.request_id <= self._latest_applied_id:
                continue
            obs = self.capture_observation(self.state, self.clock)
            if self.state.dones[0]:
                break
            self.submit(self.state.build_snapshot(self.next_id, self.control_step(), obs, 0.0), 'timeout_retry')
        return bool(expired)

    def wait_result(self):
        while True:
            result = self.poll_result()
            if result is not None:
                return result
            if self.retry_expired():
                if self.state.dones[0]:
                    return None
                continue
            with self._changed:
                if not self.jobs:
                    raise RuntimeError('No request to wait for')
                now = self.clock.now_ms
                if not self.clock.enabled:
                    result = self._take_ready()
                    if result is not None:
                        return result
                    if not self.jobs:
                        raise RuntimeError('No request to wait for')
                    deadline = min(job.deadline for job in self.jobs)
                    if deadline > now:
                        self._changed.wait((deadline - now) / 1000.0)
                    continue
                events = [job.deadline for job in self.jobs]
                for job in self.jobs:
                    if job.result is not None:
                        events.append(job.result.ready_logical_ms)
                    else:
                        events.append(max(job.arrival, self.compute_end))
                target = min(event for event in events if event > now)
            self.advance(target - now)

    def close(self):
        with self._changed:
            self._closed = True
            self.jobs.clear()
            self._changed.notify_all()
        if self._worker is not None:
            self._worker.join()


def evaluate(model_wrapper, assist, eval_env, eval_save_dir, profile_log_path, summary_path, bandwidth_trace, enable_comm_delay, chunk_waypoints=5, mode='continuous'):
    from src.vlnce_src.continue_eval import ContinuousEpisodeState
    model_wrapper.eval()
    summaries = []
    if mode == 'random':
        from src.vlnce_src.random_eval import RandomScheduler
        scheduler = RandomScheduler(args.random_seed)
    with open(profile_log_path, 'w') as fp:
        while True:
            batches = eval_env.next_minibatch()
            if batches is None:
                break
            for retry_i in range(3):
                planner = None
                try:
                    b = batches[0]
                    state = ContinuousEpisodeState(b, eval_env, assist, ignore_tiny_diff=True)
                    clock = FastEvalClock(bool(args.fast_eval), args.fast_eval_speedup)
                    clock.reset(eval_env.initial_obs_latency_ms)
                    bandwidth_trace.reset_for_episode(b['seq_name'])
                    asynchronous = mode in ('continuous', 'random')
                    planner_type = NavigationEdgePlanner if asynchronous else AsyncEdgePlanner
                    planner = planner_type(model_wrapper, bandwidth_trace, enable_comm_delay, clock)
                    capture_request = capture_observation if asynchronous else capture
                    planner.capture_observation = capture_request
                    step = 0
                    traj = []
                    index = 0
                    active = None

                    log_lock = Lock()

                    def write(r):
                        r.update(seq_names=[b['seq_name']], map_names=[b['map_name']], evaluation_method=mode)
                        with log_lock:
                            fp.write(json.dumps(r) + '\n')
                            fp.flush()
                    planner.bind_state(state, write)
                    planner.control_step = lambda: step

                    def request(reason, reuse=None):
                        obs = reuse if reuse is not None else capture_request(state, clock)
                        if not state.dones[0]:
                            snap = state.build_snapshot(0, step, obs, 0.0)
                            if mode == 'stopgo' and reason == 'initial':
                                snap.assist_notice = None
                            planner.submit(snap, reason)

                    def apply(r):
                        nonlocal traj, index, active
                        if r is None:
                            return
                        traj = copy.deepcopy(r.refined_waypoints)
                        index = 0
                        active = r
                        record = dict(record_type='decision', request_id=r.request_id, submitted_after_exec_step=r.submitted_step, result_ready_logical_ms=r.ready_logical_ms, result_applied_exec_step=step, refined_waypoints=traj, hover_wait_ms=0.0, obs_latency_ms=r.obs_latency_ms, groundingdino_latency_ms=0.0, **clock.metadata())
                        if mode != 'stopgo':
                            record.update(action_age_ms=record['logical_elapsed_ms'] - r.submitted_logical_ms, state_drift_m=float(np.linalg.norm(np.array(r.observation_pose) - state.current_sim_pose())))
                        if asynchronous:
                            record.update(result_applied_logical_ms=record['logical_elapsed_ms'], edge_service_ms=r.edge_service_ms)
                        write(record)

                    def wait():
                        start = clock.now_ms
                        apply(planner.wait_result())
                        write(dict(record_type='decision', hover_wait_ms=clock.now_ms - start, **clock.metadata()))
                    request('initial', eval_env.initial_obs_latency_ms)
                    wait()
                    initial_result = active
                    budget = int(args.max_control_steps)
                    if mode == 'stopgo':
                        budget = min(budget, int(args.maxWaypoints) * 5)
                    while not state.dones[0] and step < budget:
                        if mode != 'stopgo':
                            apply(planner.poll_result())
                        planner.retry_expired()
                        if state.dones[0]:
                            break
                        if mode == 'random':
                            decision = scheduler.decide(max(0, len(traj) - index))
                            action = decision.action
                            if decision.request:
                                reuse = getattr(state, '_boundary_obs_ms', None)
                                request('random', reuse)
                            state._boundary_obs_ms = None
                            if state.dones[0]:
                                break
                            move = decision.move
                            write(dict(record_type='scheduler', scheduler_action=action, control_step=step, buffer_remaining=max(0, len(traj) - index), duration_ms=None if move else decision.idle_ms, **clock.metadata()))
                            if not move:
                                planner.advance(decision.idle_ms)
                                continue
                        elif index >= len(traj):
                            wait()
                            continue
                        target = traj[index]
                        start = time.perf_counter()
                        eval_env.makeActionsChunk([[target]], target_idx=1)
                        timing = action_timing(eval_env, (time.perf_counter() - start) * 1000, clock.enabled)
                        clock.advance_action(timing['action_sim_time_ms'], timing['action_wall_time_ms'])
                        index += 1
                        step += 1
                        state.sync_runtime_status_from_sim()
                        obs = check_evaluation(state, model_wrapper, clock, step, write, account_ne=asynchronous) if not state.dones[0] else None
                        state._boundary_obs_ms = obs
                        flight_fields = {'action_eval_time_ms': timing['airsim_action_latency_ms']} if asynchronous else {}
                        write(dict(record_type='execution', exec_step=step, executed_waypoints=1, action_sim_time_ms=timing['action_sim_time_ms'], action_wall_time_ms=timing['action_wall_time_ms'], execution_post_obs_latency_ms=0.0, execution_post_dino_latency_ms=0.0, **flight_fields, **clock.metadata()))
                        if state.dones[0] or step >= budget:
                            break
                        if mode == 'continuous' and request_due(step, chunk_waypoints):
                            request('periodic', obs)
                            state._boundary_obs_ms = None
                        if mode == 'stopgo' and step % 5 == 0:
                            traj = []
                            index = 0
                            request('periodic', obs)
                            if not state.dones[0]:
                                wait()
                    if getattr(state, '_last_capture_step', 0) != step:
                        capture_request(state, clock)
                    state.dones[0] = True
                    state.maybe_finalize(force=True)
                    end = dict(record_type='episode_end', timing_contract='episode_logical_v2', control_steps=step, episode_latency_ms=clock.now_ms, success=bool(state.success), oracle_success=bool(state.oracle_success), collision=bool(state.collisions[0]), final_ne_m=state._calculate_distance_from_position(state.current_sim_pose()), initial_obs_latency_ms=eval_env.initial_obs_latency_ms, initial_uplink_start_logical_ms=initial_result.submitted_logical_ms, initial_uplink_latency_ms=initial_result.uplink_latency_ms, initial_uplink_payload_bits=initial_result.uplink_payload_bits, initial_uplink_payload_mb=initial_result.uplink_payload_mb, initial_uplink_bandwidth_mbps=initial_result.uplink_bandwidth_mbps, **state.cr_state.metadata(), **clock.metadata())
                    write(end)
                    summaries.append(end)
                    break
                except Exception:
                    if retry_i == 2:
                        raise
                    eval_env._changeEnv(need_change=True)
                finally:
                    if planner is not None:
                        planner.close()
    with open(summary_path, 'w') as f:
        json.dump(dict(episodes=summaries), f, indent=2)
