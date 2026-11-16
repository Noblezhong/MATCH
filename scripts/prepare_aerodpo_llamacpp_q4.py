#!/usr/bin/env python3
"""Create the Q4_K_M language GGUF used by the AeroDPO local evaluator.

The existing Q8 artifact directory remains the source of the BF16 language
model, F16 language LoRA, and BF16 multimodal projector. This tool writes only
a new Q4 language GGUF and a provenance manifest in a fresh directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_wrapper.aerodpo_llamacpp_artifacts import (
    language_quantize_command,
    language_quantized_filename,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-bf16",
        type=Path,
        default=Path(
            "/HDD1/code/AeroDPO/model_zoo/AeroDPO-llamacpp-q8_0/"
            "language-base-bf16.gguf"
        ),
    )
    parser.add_argument(
        "--llama-cpp-dir", type=Path, default=Path("/HDD1/code/llama.cpp")
    )
    parser.add_argument(
        "--quantizer",
        type=Path,
        default=Path("/HDD1/code/llama.cpp/build/bin/llama-quantize"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/HDD1/code/AeroDPO/model_zoo/AeroDPO-llamacpp-q4_k_m"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    quant_type = "Q4_K_M"
    if not args.source_bf16.is_file():
        raise FileNotFoundError(args.source_bf16)
    if not args.quantizer.is_file():
        raise FileNotFoundError(args.quantizer)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing path: {args.output_dir}")

    args.output_dir.mkdir(parents=True)
    output = args.output_dir / language_quantized_filename(quant_type)
    command = language_quantize_command(
        args.quantizer, args.source_bf16, output, quant_type
    )
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)

    llama_commit = subprocess.check_output(
        ["git", "-C", str(args.llama_cpp_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    manifest = {
        "quant_type": quant_type,
        "source_bf16": str(args.source_bf16),
        "source_bf16_sha256": sha256(args.source_bf16),
        "llama_cpp_dir": str(args.llama_cpp_dir),
        "llama_cpp_commit": llama_commit,
        "quantize_command": command,
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": sha256(output),
        "shared_lora": (
            "/HDD1/code/AeroDPO/model_zoo/AeroDPO-llamacpp-q8_0/"
            "aerodpo-language-lora-f16.gguf"
        ),
        "shared_mmproj": (
            "/HDD1/code/AeroDPO/model_zoo/AeroDPO-llamacpp-q8_0/"
            "aerodpo-mmproj-bf16.gguf"
        ),
    }
    (args.output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
