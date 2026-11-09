"""Offline configuration, prompt and scoring regression tests (no API calls)."""

import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch

import math_bench as bench

EXAMPLE = {
    line.split("=", 1)[0]: line.split("=", 1)[1]
    for line in (bench.HERE / ".env.example").read_text().splitlines()
    if line and not line.startswith("#")
}


class ConfigurationTests(unittest.TestCase):
    def config(self, **changes):
        with (
            patch.dict(os.environ, {**EXAMPLE, **changes}, clear=True),
            patch.object(bench, "load_dotenv"),
        ):
            return bench.load_config(dry_run=True)

    def test_example_prepares_nine_and_fifteen_minutes(self):
        config = self.config()
        self.assertEqual(config["order"], 9)
        self.assertEqual(config["wall_clock_min"], 15)
        self.assertEqual(config["model"], "claude-fable-5-2")
        self.assertEqual(config["provider"], "anthropic")
        self.assertEqual(config["max_output_tokens"], 16000)
        self.assertEqual(config["max_tool_calls"], 30)
        self.assertIsNone(config["target"])
        self.assertIsNone(config["previous_best"])

    def test_live_requires_key(self):
        with (
            patch.dict(os.environ, EXAMPLE, clear=True),
            patch.object(bench, "load_dotenv"),
        ):
            with self.assertRaisesRegex(SystemExit, "ANTHROPIC_API_KEY"):
                bench.load_config()

    def test_reject_invalid_budgets(self):
        for name, value in [
            ("MATH_WALL_CLOCK_MIN", "nan"),
            ("MATH_WALL_CLOCK_MIN", "inf"),
            ("MATH_ORDER", "0"),
            ("MATH_MAX_TOOL_CALLS", "-1"),
            ("MATH_MAX_OUTPUT_TOKENS", "100"),
            ("MATH_TARGET", "-1"),
            ("MATH_ORDER", "nine"),
        ]:
            with self.subTest(name=name, value=value), self.assertRaises(SystemExit):
                self.config(**{name: value})

    def test_dry_run_does_not_call_models_or_write_files(self):
        output = io.StringIO()
        with (
            patch.dict(
                os.environ,
                {**EXAMPLE, "ANTHROPIC_API_KEY": "secret-test-key"},
                clear=True,
            ),
            patch.object(bench, "load_dotenv"),
            patch.object(bench, "run_model") as anthropic_run,
            patch.object(bench, "run_openai") as openai_run,
            patch.object(bench, "run_dir") as output_dir,
            contextlib.redirect_stdout(output),
        ):
            bench.main(["--dry-run"])
        anthropic_run.assert_not_called()
        openai_run.assert_not_called()
        output_dir.assert_not_called()
        self.assertNotIn("secret-test-key", output.getvalue())
        self.assertIn("math-v4", output.getvalue())
        self.assertIn('"order": 9', output.getvalue())

    def test_prompt_has_no_unverified_record_claim(self):
        prompt = bench.build_prompt(self.config())
        self.assertIn("order-9", prompt)
        self.assertIn("15 minutes", prompt)
        self.assertIn("16,000 output tokens", prompt)
        self.assertIn("Do NOT use simulated annealing", prompt)
        self.assertIn("No comparison target", prompt)
        self.assertNotIn("best known", prompt)
        self.assertNotIn("hill-climbing", prompt)
        self.assertEqual(prompt.count("```json"), 1)

    def test_optional_comparisons_are_labeled(self):
        prompt = bench.build_prompt(
            self.config(MATH_TARGET="100", MATH_PREVIOUS_BEST="90")
        )
        self.assertIn("Configured comparison target: 100", prompt)
        self.assertIn("not verified research records", prompt)

    def test_prompt_document_matches_example(self):
        self.assertEqual(
            (bench.HERE / "docs/math/math_v4.md").read_text(),
            bench.build_prompt(self.config()),
        )


class ScoringTests(unittest.TestCase):
    def test_nine_has_no_seven_tier_or_unconfigured_target(self):
        # Identical strict rankings on each side have a single stable matching.
        n = 9
        answer = {
            "order": n,
            "men": [list(range(1, n + 1)) for _ in range(n)],
            "women": [[i + 1] * n for i in range(n)],
            "claimed_stable_matchings": 1,
        }
        run = {
            "responses": [
                {
                    "blocks": [
                        {
                            "type": "text",
                            "text": "```json\n" + json.dumps(answer) + "\n```",
                        }
                    ]
                }
            ]
        }
        result = bench.score({"order": n, "target": None}, run)
        self.assertTrue(result["valid"])
        self.assertEqual(result["verified_phi"], 1)
        self.assertIsNone(result["tier"])
        self.assertFalse(result["escalate"])
        self.assertFalse(result["honesty_flag"])


if __name__ == "__main__":
    unittest.main()
