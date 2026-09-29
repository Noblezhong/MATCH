"""Focused coordinator tests without loading the AirSim and DINO models."""

import ast
import pathlib
import time
from types import SimpleNamespace


SOURCE = pathlib.Path(__file__).resolve().parents[1] / "src/vlnce_src/match_eval.py"
METHODS = {"_step_with_ncn", "_step_ncn_recovery_hover", "_step_refresh_after_lock_ncn"}


def _coordinator_class():
    tree = ast.parse(SOURCE.read_text())
    source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MatchEnv")
    selected = [node for node in source_class.body if isinstance(node, ast.FunctionDef) and node.name in METHODS]
    test_class = ast.ClassDef(name="Coordinator", bases=[], keywords=[], body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))

    class Base:
        @staticmethod
        def _apply_result(env, result):
            env.applied.append(result.request_id)

    namespace = {
        "Dict": dict,
        "Any": object,
        "time": time,
        "args": SimpleNamespace(scheduler_req_bw_thresh_mbps=25.0),
        "DRLSchedulerEnv": Base,
    }
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["Coordinator"]


class Clock:
    enabled = True

    def __init__(self):
        self.now_ms = 0.0


class Planner:
    def __init__(self, env, results=()):
        self.env = env
        self.results = list(results)
        self.submits = 0

    def wait_result(self):
        self.env.clock.now_ms += 100.0
        return self.results.pop(0)


def _env(*, bandwidth, active=True, outage=False, results=()):
    env = _coordinator_class()()
    env.clock = Clock()
    env.state = SimpleNamespace(dones=[False])
    env.planner = Planner(env, results)
    env.target_lock = SimpleNamespace(active=False)
    env._needs_target_lock_refresh = False
    env._ncn_active = active
    env._ncn_recovery_request_id = None
    env._ncn_stale_request_ids = {1}
    env._ncn_stale_results = 0
    env._outage_held_results = []
    env._edge_latency = None
    env.scheduler_step_count = 0
    env.control_step = 8
    env.applied = []
    env.actions = []
    env.records = []
    env._sample_bandwidth = lambda: bandwidth
    env._outage_active = lambda: outage
    env._poll_edge_result_for_match = lambda: None
    env._run_ncn_action = lambda bw, reason, extra=None: env.actions.append((bw, reason, extra)) or "ncn"
    env._ncn_takeover_reason = lambda: None
    env._step_edge_only = lambda action: "ahs"
    env._submit_ncn_recovery_request = lambda bw, extra: _submit(env, extra)
    env._complete_ncn_recovery = lambda: setattr(env, "_ncn_active", False)
    env._log_ncn_event = lambda bw, extra, action_id=3: env.records.append((bw, extra, action_id)) or "logged"
    env._apply_result = lambda result: (_ for _ in ()).throw(AssertionError("TCM must not process recovery"))
    return env


def _submit(env, extra):
    env.planner.submits += 1
    env._ncn_recovery_request_id = 2
    return True


def test_stale_result_during_low_bandwidth_keeps_ncn_and_does_not_request():
    env = _env(bandwidth=10e6)
    env._poll_edge_result_for_match = lambda: SimpleNamespace(request_id=1)
    assert env._step_with_ncn(3) == "ncn"
    assert env.planner.submits == 0
    assert env._ncn_stale_results == 1
    assert env.actions[0][2]["ncn_discarded_stale_result_id"] == 1


def test_recovered_bandwidth_stops_motion_and_applies_only_fresh_result():
    class WaypointArray:
        def __len__(self):
            return 1

        def __bool__(self):
            raise ValueError("array truth value is ambiguous")

    old = SimpleNamespace(request_id=1)
    fresh = SimpleNamespace(request_id=2, refined_waypoints=WaypointArray())
    env = _env(bandwidth=25e6, results=[old, fresh])
    assert env._step_with_ncn(3) == "logged"
    assert env.planner.submits == 1
    assert env.applied == [2]
    assert env.actions == []
    assert env.control_step == 8
    assert env._ncn_active is False
    assert env._ncn_stale_results == 1
    assert env.records[0][1]["hover_wait_ms"] == 200.0
    assert env.records[0][2] == 0


def test_configured_outage_prevents_recovery_even_with_high_bandwidth():
    env = _env(bandwidth=30e6, outage=True)
    assert env._step_with_ncn(3) == "ncn"
    assert env.planner.submits == 0


def test_normal_ahs_path_does_not_preemptively_poll_and_lose_result():
    env = _env(bandwidth=20e6, active=False)
    env._poll_edge_result_for_match = lambda: (_ for _ in ()).throw(AssertionError("premature poll"))
    assert env._step_with_ncn(0) == "ahs"


def test_tcm_lock_refresh_hovers_without_entering_ncn():
    env = _env(bandwidth=25e6, active=False)
    env._needs_target_lock_refresh = True
    env._step_refresh_after_lock = lambda bw, start, logical: ("hover_refresh", bw)
    assert env._step_with_ncn(3) == ("hover_refresh", 25e6)
    assert env.scheduler_step_count == 1
    assert env.actions == []


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
