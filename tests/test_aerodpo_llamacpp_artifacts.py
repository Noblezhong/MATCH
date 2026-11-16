import unittest

import torch

from src.model_wrapper.aerodpo_llamacpp_artifacts import partition_adapter_tensors


class AeroDPOLlamaCppArtifactTest(unittest.TestCase):
    def test_partition_keeps_language_lora_and_projector_separate(self):
        tensors = {
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 4),
            "base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2),
            "base_model.model.model.visual.merger.linear_fc1.weight": torch.ones(4, 4),
            "base_model.model.model.visual.merger.norm.bias": torch.ones(4),
        }

        lora, projector = partition_adapter_tensors(tensors)

        self.assertEqual(len(lora), 2)
        self.assertEqual(
            set(projector),
            {
                "visual.merger.linear_fc1.weight",
                "visual.merger.norm.bias",
            },
        )
        self.assertTrue(
            torch.equal(projector["visual.merger.linear_fc1.weight"], torch.ones(4, 4))
        )

    def test_partition_rejects_tensor_outside_lora_or_projector(self):
        tensors = {"base_model.model.model.unexpected.weight": torch.ones(2, 2)}

        with self.assertRaisesRegex(ValueError, "unexpected adapter tensor"):
            partition_adapter_tensors(tensors)


if __name__ == "__main__":
    unittest.main()
