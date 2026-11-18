"""Common episode collision policy for all MATCH evaluators."""
from threading import RLock

COLLISION_POLICY = "episode_collision_v1"


class AirSimCollisionTracker:
    """Per-client/vehicle lifetime is one episode; sensors may be recreated."""
    def __init__(self):
        self.initial_raw = None
        self.cleared = False
        self.airsim_seen = False
        self.timeout_seen = False
        self.lock = RLock()

    def observe(self, raw):
        with self.lock:
            raw = bool(raw)
            if self.initial_raw is None:
                self.initial_raw = raw
                self.cleared = not raw
                valid = False  # baseline, before navigation
            else:
                if not raw:
                    self.cleared = True
                valid = raw and self.cleared
            self.airsim_seen |= valid
            return {
                "valid_has_collided": valid,
                "airsim_initial_has_collided": self.initial_raw,
                "airsim_cleared_since_start": self.cleared,
                "valid_airsim_seen": self.airsim_seen,
                "waypoint_timeout_seen": self.timeout_seen,
                "collision_policy": COLLISION_POLICY,
            }

    def record_timeout(self, collision):
        with self.lock:
            self.timeout_seen = True
            collision["waypoint_timeout_seen"] = True


def valid_airsim_collision(state):
    return bool(state.get("collision", {}).get("valid_has_collided", False))


class EpisodeCollisionState:
    def __init__(self):
        self.airsim_collision = False
        self.waypoint_timeout = False
        self.depth_collision = False
        self.cr_collision = False

    def observe(self, observations):
        for observation in observations:
            collision = observation.get("sensors", {}).get("state", {}).get("collision", {})
            self.airsim_collision |= bool(collision.get("valid_airsim_seen", False))
            self.waypoint_timeout |= bool(collision.get("waypoint_timeout_seen", False))
            self.depth_collision |= bool(collision.get("depth_collision", False))
        self.cr_collision |= self.airsim_collision or self.waypoint_timeout or self.depth_collision
        return self.cr_collision

    def metadata(self):
        return {
            "collision_policy": COLLISION_POLICY,
            "cr_collision": bool(self.cr_collision),
            "cr_airsim_collision": bool(self.airsim_collision),
            "cr_waypoint_timeout": bool(self.waypoint_timeout),
            "cr_depth_collision": bool(self.depth_collision),
        }
