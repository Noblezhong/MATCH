"""AeroVLA's native OpenVLA policy adapted to TravelUAV's AeroDPO RGB-D view pair."""
import re
import time

import cv2
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from scipy.spatial.transform import Rotation
from transformers import AutoImageProcessor, AutoModelForVision2Seq, AutoTokenizer

from src.model_wrapper.base_model import BaseModelWrapper


class AeroVLAEvalModelWrapper(BaseModelWrapper):
    """Preserve AeroVLA inference while reading AeroDPO's front/down observations."""

    resolution = 224
    num_bins = 99
    norm_stats = {
        "forward": {"min": 0.0, "max": 5.0},
        "down": {"min": -5.0, "max": 5.0},
        "yaw": {"min": -1.1, "max": 1.1},
    }

    def __init__(self, model_args):
        super().__init__()
        base_model_path = model_args.base_model_path
        adapter_path = model_args.model_path
        if not base_model_path:
            raise ValueError("--base_model_path must point to the OpenVLA-7B base")
        if not adapter_path:
            raise ValueError("--model_path must point to the AeroVLA LoRA adapter")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[AeroVLA Edge] Loading BASE from: {base_model_path}", flush=True)
        print(f"[AeroVLA Edge] Loading ADAPTER from: {adapter_path}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.image_processor = AutoImageProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        base_model.resize_token_embeddings(len(self.tokenizer))
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.to(self.device)
        self.model.eval()

    def _sync_cuda(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def eval(self):
        self.model.eval()

    @staticmethod
    def semantic_direction(current_state, target_position):
        position = np.asarray(current_state["position"], dtype=np.float64)
        raw_orientation = current_state["orientation"]
        if isinstance(raw_orientation, dict):
            quaternion = [raw_orientation.get("x", 0.0), raw_orientation.get("y", 0.0),
                         raw_orientation.get("z", 0.0), raw_orientation.get("w", 1.0)]
        else:
            quaternion = raw_orientation
        world_vector = np.asarray(target_position, dtype=np.float64) - position
        if float(np.linalg.norm(world_vector[:2])) < 0.01:
            return ""
        body_vector = Rotation.from_quat(quaternion).inv().apply(world_vector)
        angle = float(np.degrees(np.arctan2(body_vector[1], body_vector[0])))
        if -15 <= angle <= 15:
            return "straight ahead "
        if 15 < angle <= 60:
            return "forward-right "
        if 60 < angle <= 120:
            return "to your right "
        if 120 < angle <= 180:
            return "to your right rear "
        if -60 <= angle < -15:
            return "forward-left "
        if -120 <= angle < -60:
            return "to your left "
        if -180 <= angle < -120:
            return "to your left rear "
        return ""

    @staticmethod
    def _object_description(instruction):
        return instruction.split("degrees from you.", 1)[1].split(" Please control", 1)[0].strip()

    def prepare_inputs(self, episodes, target_positions, instructions):
        prompts, pixel_values_list = [], []
        for index, episode in enumerate(episodes):
            latest = episode[-1]
            rgb_views = latest["rgb"]
            front_rgb = cv2.cvtColor(rgb_views[0], cv2.COLOR_BGR2RGB)
            down_rgb = cv2.cvtColor(rgb_views[1], cv2.COLOR_BGR2RGB)
            front = Image.fromarray(front_rgb).resize(
                (self.resolution, self.resolution), resample=Image.Resampling.BICUBIC
            )
            down = Image.fromarray(down_rgb).resize(
                (self.resolution, self.resolution), resample=Image.Resampling.BICUBIC
            )
            mosaic = Image.new("RGB", (self.resolution, self.resolution * 2), (0, 0, 0))
            mosaic.paste(front, (0, 0))
            mosaic.paste(down, (0, self.resolution))
            direction = self.semantic_direction(
                latest["sensors"]["state"], target_positions[index]
            )
            description = self._object_description(instructions[index])
            prompts.append(f"<image>\nFly {direction}and find the target. {description}\nAction: ")
            pixel_values = self.image_processor(images=mosaic, return_tensors="pt")["pixel_values"].squeeze(0)
            pixel_values_list.append(pixel_values)
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        pixel_values = torch.stack(pixel_values_list).to(self.device)
        if hasattr(self.model, "dtype"):
            pixel_values = pixel_values.to(self.model.dtype)
        inputs["pixel_values"] = pixel_values
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        return inputs, prompts

    def parse_action_from_text(self, text):
        output_part = text.split("Action:")[-1]
        matches = re.findall(r"\d+", output_part)
        values = [0.0, 0.0, 0.0]
        if len(matches) >= 3:
            for index, axis in enumerate(("forward", "down", "yaw")):
                low, high = self.norm_stats[axis]["min"], self.norm_stats[axis]["max"]
                bin_value = max(0, min(self.num_bins - 1, int(matches[-3 + index])))
                values[index] = (bin_value / (self.num_bins - 1)) * (high - low) + low
        action = {"fwd": values[0], "down": values[1], "yaw": values[2]}
        should_stop = "LAND" in text or "<LAND>" in text or (
            action["fwd"] < 0.01 and abs(action["down"]) < 0.01 and abs(action["yaw"]) < 0.01
        )
        return action, should_stop

    def run_profiled(self, inputs):
        with torch.inference_mode():
            self._sync_cuda()
            start = time.perf_counter()
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                eos_token_id=[self.tokenizer.eos_token_id],
            )
            self._sync_cuda()
            latency_ms = (time.perf_counter() - start) * 1000.0
        actions, stops, texts = [], [], []
        for output_ids in generated_ids:
            text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
            action, should_stop = self.parse_action_from_text(text)
            actions.append(action)
            stops.append(should_stop)
            texts.append(text)
        return actions, stops, {
            "model_inference_latency_ms": latency_ms,
            "model_output_text": texts,
        }
