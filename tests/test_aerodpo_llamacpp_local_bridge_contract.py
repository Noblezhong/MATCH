import unittest
from pathlib import Path

import numpy as np

from src.model_wrapper.aerodpo_llamacpp_local_wrapper import AeroDPOLlamaCppLocalBridge


class _Free:
    def __init__(self):
        self.released = []

    def __call__(self, handle):
        self.released.append(handle)


class _Lib:
    def __init__(self):
        self.aerodpo_llamacpp_local_free = _Free()


class LocalBridgeContractTest(unittest.TestCase):
    def test_bridge_rejects_non_mosaic_shape_before_native_call(self):
        bridge = AeroDPOLlamaCppLocalBridge.__new__(AeroDPOLlamaCppLocalBridge)
        with self.assertRaisesRegex(ValueError, r"expected RGB"):
            bridge.generate(np.zeros((384, 384, 3), dtype=np.uint8), "prompt")

    def test_close_releases_handle_once(self):
        bridge = AeroDPOLlamaCppLocalBridge.__new__(AeroDPOLlamaCppLocalBridge)
        bridge.lib = _Lib()
        bridge.handle = 123
        bridge.close()
        bridge.close()
        self.assertEqual(bridge.lib.aerodpo_llamacpp_local_free.released, [123])

    def test_http_backend_files_are_absent(self):
        self.assertFalse(Path("scripts/aerodpo_llamacpp_server.sh").exists())
        self.assertFalse(
            Path("src/model_wrapper/aerodpo_llamacpp_eval_wrapper.py").exists()
        )


if __name__ == "__main__":
    unittest.main()

