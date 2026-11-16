import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from src.model_wrapper.aerodpo_llamacpp_local_wrapper import (
    AeroDPOLlamaCppLocalEvalModelWrapper,
)


class _Bridge:
    def generate(self, rgb, prompt, max_tokens):
        assert rgb.dtype == np.uint8
        assert rgb.flags.c_contiguous
        assert rgb.shape == (768, 384, 3)
        assert prompt.endswith("Action: ")
        assert max_tokens == 20
        return "43 10 23", {
            "native_bridge_latency_ms": 12.0,
            "native_vision_encode_latency_ms": 4.0,
            "native_prefill_latency_ms": 5.0,
            "native_decode_latency_ms": 3.0,
        }


class LocalWrapperTest(unittest.TestCase):
    def test_run_profiled_returns_native_action_and_profile(self):
        wrapper = AeroDPOLlamaCppLocalEvalModelWrapper.__new__(
            AeroDPOLlamaCppLocalEvalModelWrapper
        )
        wrapper.bridge = _Bridge()

        actions, stops, profile = wrapper.run_profiled(
            [{
                "image": Image.new("RGB", (384, 768)),
                "prompt": "Fly straight ahead and find the target. object\nAction: ",
            }]
        )

        self.assertFalse(stops[0])
        self.assertAlmostEqual(actions[0]["fwd"], 43 / 98 * 5)
        self.assertEqual(profile["model_output_text"], ["43 10 23"])
        self.assertEqual(profile["native_bridge_latency_ms"], 12.0)
        self.assertGreater(profile["model_inference_latency_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
