#!/usr/bin/env python3
"""Verify that splitting the AeroDPO adapter preserves PyTorch behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

from src.model_wrapper.aerodpo_eval_wrapper import AeroDPOEvalModelWrapper
from src.model_wrapper.aerodpo_llamacpp_artifacts import partition_adapter_tensors
from src.vlnce_src.aerodpo_eval_contract import parse_aerodpo_action_text
from safetensors.torch import load_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-dir", type=Path, required=True)
    parser.add_argument("--original-adapter-dir", type=Path, required=True)
    parser.add_argument("--language-adapter-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    return parser.parse_args()


def make_inputs(processor, index: int, device: torch.device):
    rng = np.random.default_rng(20260828 + index)
    pixels = rng.integers(0, 256, size=(768, 384, 3), dtype=np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    prompt = f"Fly straight ahead and find the target. fixed test object {index}.\nAction: "
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info([messages])
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding="longest",
        return_tensors="pt",
    )
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    return {key: value.to(device) for key, value in inputs.items()}


def load_policy(base_dir: Path, adapter_dir: Path | None, projector=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = AutoModelForImageTextToText.from_pretrained(
        base_dir,
        dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    if projector is not None:
        missing, unexpected = base.load_state_dict(projector, strict=False)
        if unexpected:
            raise RuntimeError(f"unexpected projector keys: {unexpected}")
        if len(missing) <= len(projector):
            raise RuntimeError("projector overrides were not applied")
    policy = (
        PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        if adapter_dir is not None
        else base
    )
    policy.to(device)
    policy.eval()
    return policy, device


def run_policy(policy, processor, device, samples):
    outputs = []
    with torch.inference_mode():
        for index in range(samples):
            inputs = make_inputs(processor, index, device)
            generated = policy.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                use_cache=True,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            input_length = inputs["input_ids"].shape[1]
            text = processor.decode(
                generated[0][input_length:], skip_special_tokens=True
            ).strip()
            action, stop = parse_aerodpo_action_text(text)
            outputs.append({
                "index": index,
                "token_ids": generated[0][input_length:].detach().cpu().tolist(),
                "text": text,
                "action": action,
                "stop": stop,
            })
    return outputs


def main():
    args = parse_args()
    processor = AutoProcessor.from_pretrained(
        args.base_model_dir, trust_remote_code=True
    )
    processor.image_processor.max_pixels = 384 * 384 * 2
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    reference, device = load_policy(
        args.base_model_dir, args.original_adapter_dir
    )
    reference_outputs = run_policy(reference, processor, device, args.samples)
    del reference
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adapter_tensors = load_file(
        str(args.original_adapter_dir / "adapter_model.safetensors"),
        device="cpu",
    )
    _, projector = partition_adapter_tensors(adapter_tensors)
    base_projector = {f"model.{name}": tensor for name, tensor in projector.items()}
    candidate, device = load_policy(
        args.base_model_dir, args.language_adapter_dir, base_projector
    )
    candidate_outputs = run_policy(candidate, processor, device, args.samples)
    del candidate
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    base_only, device = load_policy(args.base_model_dir, None, base_projector)
    base_outputs = run_policy(base_only, processor, device, args.samples)

    mismatches = []
    for reference_item, candidate_item in zip(reference_outputs, candidate_outputs):
        if reference_item != candidate_item:
            mismatches.append({"reference": reference_item, "candidate": candidate_item})
    report = {
        "samples": args.samples,
        "matching_samples": args.samples - len(mismatches),
        "mismatch_count": len(mismatches),
        "action_match_rate": (
            (args.samples - len(mismatches)) / args.samples if args.samples else 0.0
        ),
        "reference_outputs": reference_outputs,
        "candidate_outputs": candidate_outputs,
        "base_projector_outputs": base_outputs,
        "mismatches": mismatches,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
