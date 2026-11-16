import unittest

from src.model_wrapper.aerodpo_llamacpp_artifacts import language_only_adapter_config


class AeroDPOLlamaCppAdapterConfigTest(unittest.TestCase):
    def test_language_only_adapter_removes_projector_module_save(self):
        config = {
            "peft_type": "LORA",
            "modules_to_save": ["visual.merger"],
            "lora_alpha": 128,
        }

        result = language_only_adapter_config(config)

        self.assertEqual(result["modules_to_save"], [])
        self.assertEqual(result["lora_alpha"], 128)
        self.assertEqual(config["modules_to_save"], ["visual.merger"])


if __name__ == "__main__":
    unittest.main()
