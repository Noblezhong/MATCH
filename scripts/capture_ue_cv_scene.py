#!/usr/bin/env python3
"""Capture a high-resolution UE image from an isolated AirSim CV instance.

It never contacts the benchmark AirSim service: this process launches its own
UE instance, RPC settings, and process group, then shuts down only that group.

Example:
  conda run -n llamauav python scripts/capture_ue_cv_scene.py --overwrite
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

import airsim
import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_SCRIPT = Path("/HDD2/AeroDuo_envs/closeloop_envs/NYCEnvironmentMegapa.sh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-script", type=Path, default=DEFAULT_SCENE_SCRIPT)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "scene_captures" / "nyc_cv_1920x1080.png")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--requested-port", type=int, default=32001)
    parser.add_argument("--fallback-port", type=int, default=41451,
                        help="Fallback for packaged AirSim builds that ignore ApiServerPort.")
    parser.add_argument("--ready-timeout", type=float, default=90.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fov", type=float, default=90.0)
    parser.add_argument("--x", type=float, default=619.86798)
    parser.add_argument("--y", type=float, default=48.84400)
    parser.add_argument("--z", type=float, default=-25.55400)
    parser.add_argument("--pitch", type=float, default=0.30)
    parser.add_argument("--roll", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.198)
    parser.add_argument("--settings-path", type=Path,
                        default=Path("/tmp/aerodpo_ue_cv_capture_settings.json"))
    parser.add_argument("--ue-log-path", type=Path,
                        default=Path("/tmp/aerodpo_ue_cv_capture.log"))
    return parser.parse_args()


def is_port_open(port: int) -> bool:
    sock = socket.socket()
    sock.settimeout(1.0)
    try:
        return sock.connect_ex(("127.0.0.1", port)) == 0
    finally:
        sock.close()


def wait_for_client(process: subprocess.Popen[bytes], ports: tuple[int, ...], timeout_s: float) -> tuple[airsim.VehicleClient, int]:
    deadline = time.monotonic() + timeout_s
    next_status = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"temporary UE exited early with code {process.returncode}")
        for port in ports:
            try:
                client = airsim.VehicleClient(ip="127.0.0.1", port=port, timeout_value=3)
                client.confirmConnection()
                return client, port
            except Exception:
                pass
        if time.monotonic() >= next_status:
            print(f"waiting for isolated AirSim RPC on ports {ports}", flush=True)
            next_status += 15.0
        time.sleep(1.0)
    raise TimeoutError(f"isolated AirSim was not ready within {timeout_s:.0f} seconds")


def stop_process_group(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=10)


def capture(client: airsim.VehicleClient, args: argparse.Namespace) -> dict[str, object]:
    pose = airsim.Pose(
        airsim.Vector3r(args.x, args.y, args.z),
        airsim.to_quaternion(args.pitch, args.roll, args.yaw),
    )
    client.simSetVehiclePose(pose, True)
    time.sleep(2.0)
    response = None
    for _ in range(2):
        image = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])[0]
        if image.width == args.width and image.height == args.height and image.image_data_uint8:
            response = image
            break
        time.sleep(1.0)
    if response is None:
        raise RuntimeError(f"AirSim did not return a {args.width}x{args.height} Scene image")
    pixels = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    expected = response.width * response.height * 3
    if pixels.size != expected:
        raise RuntimeError(f"unexpected raw image length {pixels.size}, expected {expected}")
    image_bgr = pixels.reshape(response.height, response.width, 3)
    mean, std = float(image_bgr.mean()), float(image_bgr.std())
    if mean < 2.0 or std < 1.0:
        raise RuntimeError(f"invalid image statistics: mean={mean:.2f}, std={std:.2f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite")
    if not cv2.imwrite(str(args.output), image_bgr):
        raise RuntimeError(f"failed to write {args.output}")
    return {"output": str(args.output), "width": response.width, "height": response.height,
            "bytes": args.output.stat().st_size, "pixel_mean": mean, "pixel_std": std}


def main() -> None:
    args = parse_args()
    logging.getLogger("tornado.general").setLevel(logging.ERROR)
    if not args.scene_script.is_file():
        raise FileNotFoundError(args.scene_script)
    ports = tuple(dict.fromkeys((args.requested_port, args.fallback_port)))
    for port in ports:
        if is_port_open(port):
            raise RuntimeError(f"refusing to use occupied port {port}")
    settings = {
        "SettingsVersion": 1.2, "SimMode": "ComputerVision", "ViewMode": "NoDisplay",
        "ApiServerPort": args.requested_port, "ClockSpeed": 1,
        "CameraDefaults": {"CaptureSettings": [{"ImageType": 0, "Width": args.width,
            "Height": args.height, "FOV_Degrees": args.fov, "AutoExposureSpeed": 100,
            "MotionBlurAmount": 0}]},
    }
    args.settings_path.write_text(json.dumps(settings), encoding="utf-8")
    log_handle = args.ue_log_path.open("wb")
    process: subprocess.Popen[bytes] | None = None
    try:
        command = ["setsid", "bash", str(args.scene_script), "-RenderOffscreen", "-NoSound",
                   "-NoVSync", f"-GraphicsAdapter={args.gpu}", f"-settings={args.settings_path}"]
        process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT)
        print(json.dumps({"temporary_ue_pgid": process.pid, "ports": ports}), flush=True)
        client, connected_port = wait_for_client(process, ports, args.ready_timeout)
        result = capture(client, args)
        result["connected_port"] = connected_port
        print(json.dumps(result), flush=True)
    finally:
        stop_process_group(process)
        log_handle.close()
        args.settings_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
