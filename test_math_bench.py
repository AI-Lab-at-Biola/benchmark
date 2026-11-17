import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from dotenv import dotenv_values

import math_bench


class OutputTests(unittest.TestCase):
    def test_output_root(self):
        self.assertEqual(math_bench.OUTPUT_DIR, math_bench.HERE / "out" / "math")

    def test_model_directories(self):
        started = datetime(2026, 11, 17, 12, 30, 45)
        with tempfile.TemporaryDirectory() as root, patch.object(
            math_bench, "OUTPUT_DIR", Path(root)
        ):
            for model in ("gpt-6.2-sol", "gpt-6.2-astra"):
                with self.subTest(model=model):
                    path = math_bench.run_dir(started, model)
                    self.assertEqual(
                        path, Path(root) / model / f"2026-11-17_123045_{model}"
                    )
                    self.assertTrue(path.is_dir())
                    # Same-time runs must not silently overwrite existing results.
                    with self.assertRaises(FileExistsError):
                        math_bench.run_dir(started, model)
            path = math_bench.run_dir(started, "../provider/model")
            self.assertTrue(path.resolve().is_relative_to(Path(root).resolve()))
            self.assertEqual(path.parent.name, "_provider_model")

    def test_new_models_load_from_example(self):
        config = dotenv_values(math_bench.HERE / ".env.example")
        config["OPENAI_API_KEY"] = "test-key"
        for model in ("gpt-6.2-sol", "gpt-6.2-astra"):
            with self.subTest(model=model), patch.dict(
                "os.environ", {**config, "MATH_MODEL": model}, clear=True
            ), patch("math_bench.load_dotenv"):
                loaded = math_bench.load_config()
                self.assertEqual(loaded["model"], model)
                self.assertEqual(loaded["provider"], "openai")
                self.assertEqual(loaded["effort"], "max")
