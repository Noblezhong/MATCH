#!/usr/bin/env python3
"""Capture a real third-person UAV frame from an isolated AirSim instance.

The script launches its own Unreal process and RPC port. It never connects to
the benchmark simulator used by an evaluation job.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import airsim
import cv2
import numpy as np


DEFAULT_SCENE = Path("/HDD2/AeroDuo_envs/closeloop_envs/NYCEnvironmentMegapa.sh")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-script", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=32011)
    parser.add_argument("--fallback-port", type=int, default=41451)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fov", type=float, default=82.0)
    parser.add_argument("--x", type=float, default=619.868)
    parser.add_argument("--y", type=float, default=48.844)
    parser.add_argument("--z", type=float, default=-25.554)
    parser.add_argument("--yaw", type=float, default=0.198)
    parser.add_argument("--distance", type=float, default=7.5)
    parser.add_argument("--height-offset", type=float, default=3.0)
    parser.add_argument("--ready-timeout", type=float, default=120.0)
    return parser.parse_args()


def port_open(port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(0.5)
    try:
        return sock.connect_ex(("127.0.0.1", port)) == 0
    finally:
        sock.close()


def wait_for_airsim(process: subprocess.Popen[bytes], ports: tuple[int, ...], timeout: float) -> airsim.MultirotorClient:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Unreal exited with code {process.returncode}")
        for port in ports:
            try:
                client = airsim.MultirotorClient(ip="127.0.0.1", port=port, timeout_value=3)
                client.confirmConnection()
                return client
            except Exception:
                pass
        time.sleep(1.0)
    raise TimeoutError("isolated AirSim instance did not become ready")


def stop_group(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def look_at_quaternion(camera_x: float, camera_y: float, camera_z: float,
                       target_x: float, target_y: float, target_z: float):
    dx, dy, dz = target_x - camera_x, target_y - camera_y, target_z - camera_z
    yaw = math.atan2(dy, dx)
    pitch = math.atan2(-dz, math.hypot(dx, dy))
    return airsim.to_quaternion(pitch, 0.0, yaw)


def main() -> None:
    args = arguments()
    if not args.scene_script.is_file():
        raise FileNotFoundError(args.scene_script)
    ports = tuple(dict.fromkeys((args.port, args.fallback_port)))
    for port in ports:
        if port_open(port):
            raise RuntimeError(f"refusing to use occupied port {port}")

    capture = [{"ImageType": 0, "Width": args.width, "Height": args.height,
                "FOV_Degrees": args.fov, "AutoExposureSpeed": 100,
                "MotionBlurAmount": 0}]
    settings = {
        "SettingsVersion": 1.2,
        "SimMode": "Multirotor",
        "ViewMode": "NoDisplay",
        "ApiServerPort": args.port,
        "ClockSpeed": 1,
        "Vehicles": {
            "Drone_1": {
                "VehicleType": "SimpleFlight",
                "UseSerial": False,
                "LockStep": True,
                "AutoCreate": True,
                "X": 0, "Y": 0, "Z": 0,
                "Roll": 0, "Pitch": 0, "Yaw": 0,
                "Cameras": {"FrontCamera": {"CaptureSettings": capture}},
            }
        },
        "ExternalCameras": {
            "ChaseCamera": {
                "X": 0, "Y": 0, "Z": 0,
                "Roll": 0, "Pitch": 0, "Yaw": 0,
                "CaptureSettings": capture,
            }
        },
    }

    settings_path = Path(f"/tmp/match_third_person_{args.port}.json")
    log_path = Path(f"/tmp/match_third_person_{args.port}.log")
    settings_path.write_text(json.dumps(settings), encoding="utf-8")
    log_handle = log_path.open("wb")
    process = None
    try:
        command = ["setsid", "bash", str(args.scene_script), "-RenderOffscreen",
                   "-NoSound", "-NoVSync", f"-GraphicsAdapter={args.gpu}",
                   f"-settings={settings_path}"]
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
        client = wait_for_airsim(process, ports, args.ready_timeout)

        vehicle_pose = airsim.Pose(
            airsim.Vector3r(args.x, args.y, args.z),
            airsim.to_quaternion(0.0, 0.0, args.yaw),
        )
        client.simSetVehiclePose(vehicle_pose, True, vehicle_name="Drone_1")

        camera_x = args.x - args.distance * math.cos(args.yaw)
        camera_y = args.y - args.distance * math.sin(args.yaw)
        camera_z = args.z - args.height_offset
        camera_pose = airsim.Pose(
            airsim.Vector3r(camera_x, camera_y, camera_z),
            look_at_quaternion(camera_x, camera_y, camera_z,
                               args.x + 5.0 * math.cos(args.yaw),
                               args.y + 5.0 * math.sin(args.yaw), args.z),
        )
        client.simSetCameraPose("ChaseCamera", camera_pose, external=True)
        time.sleep(3.0)
        response = client.simGetImages(
            [airsim.ImageRequest("ChaseCamera", airsim.ImageType.Scene, False, False)],
            external=True,
        )[0]
        pixels = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
        expected = response.width * response.height * 3
        if pixels.size != expected:
            raise RuntimeError(f"unexpected image size {pixels.size}; expected {expected}")
        frame = pixels.reshape(response.height, response.width, 3)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), frame):
            raise RuntimeError(f"failed to write {args.output}")
        print(json.dumps({"output": str(args.output), "width": response.width,
                          "height": response.height, "bytes": args.output.stat().st_size}))
    finally:
        stop_group(process)
        log_handle.close()
        settings_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
