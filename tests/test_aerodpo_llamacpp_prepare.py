import unittest
from pathlib import Path

from src.model_wrapper.aerodpo_llamacpp_artifacts import (
    conversion_commands,
    language_quantize_command,
    language_quantized_filename,
)


class AeroDPOLlamaCppPrepareTest(unittest.TestCase):
    def test_conversion_commands_keep_mmproj_and_lora_high_precision(self):
        commands = conversion_commands(
            llama_cpp_dir=Path("/llama.cpp"),
            base_dir=Path("/base"),
            language_adapter_dir=Path("/adapter"),
            staged_dir=Path("/staged"),
            output_dir=Path("/output"),
        )

        self.assertIn("--outtype bf16", commands["text_bf16"])
        self.assertTrue(commands["text_q8"].endswith(" Q8_0"))
        self.assertIn("--outtype f16", commands["lora_f16"])
        self.assertIn("--mmproj", commands["mmproj_bf16"])
        self.assertIn("--outtype bf16", commands["mmproj_bf16"])

    def test_q4_quantization_starts_from_bf16_and_uses_canonical_name(self):
        source = Path("/artifacts/language-base-bf16.gguf")
        output = Path("/q4/language-base-q4_k_m.gguf")
        self.assertEqual(language_quantized_filename("Q4_K_M"), output.name)
        self.assertEqual(
            language_quantize_command(Path("/llama-quantize"), source, output, "Q4_K_M"),
            ["/llama-quantize", str(source), str(output), "Q4_K_M"],
        )


if __name__ == "__main__":
    unittest.main()
