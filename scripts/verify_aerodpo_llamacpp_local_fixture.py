#!/usr/bin/env python3
"""Verify process-local llama.cpp output against the checked-in fixture."""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.model_wrapper.aerodpo_llamacpp_local_wrapper import AeroDPOLlamaCppLocalBridge
from src.vlnce_src.aerodpo_eval_contract import parse_aerodpo_action_text

DEFAULT_ARTIFACT_DIR = Path("/HDD1/code/AeroDPO/model_zoo/AeroDPO-llamacpp-q8_0")
DEFAULT_BRIDGE = Path(
    "native/aerodpo_llamacpp_local/build/libaerodpo_llamacpp_local.so"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tests/fixtures/aerodpo_llamacpp_bf16_reference.json"),
    )
    parser.add_argument(
        "--text-model",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "language-base-bf16.gguf",
    )
    parser.add_argument(
        "--lora-model",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "aerodpo-language-lora-f16.gguf",
    )
    parser.add_argument(
        "--mmproj-model",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "aerodpo-mmproj-bf16.gguf",
    )
    parser.add_argument("--bridge", type=Path, default=DEFAULT_BRIDGE)
    parser.add_argument(
        "--allow-different-valid-actions",
        action="store_true",
        help="for lossy quantization: require legal action/LAND, not BF16 equality",
    )
    args = parser.parse_args()

    expected = json.loads(args.fixture.read_text(encoding="utf-8"))["llamacpp_outputs"]
    bridge = AeroDPOLlamaCppLocalBridge(
        args.text_model, args.lora_model, args.mmproj_model, args.bridge
    )
    actual = []
    try:
        for item in expected:
            index = int(item["index"])
            rgb = np.random.default_rng(20260828 + index).integers(
                0, 256, size=(768, 384, 3), dtype=np.uint8
            )
            prompt = (
                "Fly straight ahead and find the target. "
                f"fixed test object {index}.\nAction: "
            )
            text, _ = bridge.generate(rgb, prompt)
            action, stop = parse_aerodpo_action_text(text)
            numbers = re.findall(r"\d+", text)
            exact_match = (
                text == item["text"]
                and action == item["action"]
                and stop == item["stop"]
            )
            valid = "LAND" in text or len(numbers) >= 3
            actual.append(
                {
                    "index": index,
                    "text": text,
                    "action": action,
                    "stop": stop,
                    "match": valid if args.allow_different_valid_actions else exact_match,
                    "exact_match": exact_match,
                    "valid_action_or_land": valid,
                }
            )
    finally:
        bridge.close()

    passed = all(row["match"] for row in actual)
    print(json.dumps({"passed": passed, "samples": actual}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

