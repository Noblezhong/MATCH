import unittest

import torch

from src.model_wrapper.aerodpo_llamacpp_artifacts import projector_overrides_for_base


class AeroDPOLlamaCppProjectorTest(unittest.TestCase):
    def test_projector_overrides_match_base_checkpoint_keys(self):
        projector = {
            "visual.merger.linear_fc1.weight": torch.ones(4, 4),
        }

        overrides = projector_overrides_for_base(projector)

        self.assertEqual(
            set(overrides),
            {
                "model.visual.merger.linear_fc1.weight",
            },
        )


if __name__ == "__main__":
    unittest.main()
