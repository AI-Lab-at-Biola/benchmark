# benchmark

The Biola AI LAB's independent LLM benchmark gives an Anthropic or OpenAI model a code sandbox to find stable marriage instances with many stable matchings, then independently verifies and logs the result.

## Fable preparation

The new `math-v4` prompt uses **n=9** and a **15-minute model-run budget**, with concise output and code-verified final JSON. The host independently recounts all 9! possible matchings after the model run; verification and file writing are outside that model-run budget. Historical n=7 score tiers do not apply to n=9.

Planned cadence: **four runs on day 0, day 2, day 4, and day 6** of one week. No start date, scheduler, or automatic paid run is configured. Run each separately once timing and model access are confirmed.

## Setup and safe preview

```bash
cp .env.example .env
uv run math_bench.py --dry-run
```

The preview needs no API key, constructs no API client, writes no run files, and prints no credentials. Dependencies must still be installed.

Before a live run:

1. The example selects `anthropic`, `claude-fable-5-2`, and effort `high`, per the [official Fable 5.2 documentation](https://platform.claude.com/docs/en/models/fable-5-2/overview). Confirm your account has model and code-execution access before running.
2. Add the matching provider API key to `.env`. Do not commit it.
3. Review the adjustable starting limits: **16,000 output tokens and 30 tool calls** per run. These reduce the prior 400,000-token/150-call example, but are not a user-approved dollar budget. There is no input-token spending cap; conversation replay, caching, and hosted tools can incur additional charges. Provider calls may consume usage before a stop is observed. A client-side deadline does not guarantee immediate provider-side cancellation or a final bill, so counts are not a billing guarantee.
4. Optionally supply `MATH_TARGET` and `MATH_PREVIOUS_BEST` as n=9 comparison values. Blank values are allowed. They are not asserted to be current world records. Exceeding a configured target flags review, not a proven research breakthrough.

## Run and results

```bash
uv run math_bench.py
```

Results go to `out/math/<datetime>_<model>/` as `run.json` and `run.md`. Updates report cumulative API-reported input/output token usage. For Anthropic, displayed input includes cache reads and writes; the raw usage fields remain separately recorded. Interrupted responses can leave incomplete lower bounds, not final billing totals. Preserve the config, requested/served model, prompt version/hash, verified count, elapsed time, tool calls, and token usage for each run. Keep settings consistent across the four runs when comparing outcomes.

```bash
uv run python -m unittest test_model_calls.py test_math_bench.py
```

See [the v4 prompt](docs/math/math_v4.md). Earlier prompt documents are historical and unchanged.

## Roadmap

- Design Benchmark
- Theology Benchmark
