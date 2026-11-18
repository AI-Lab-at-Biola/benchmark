# benchmark

This is the Biola AI LAB's independent LLM benchmark. Currently the benchmark gives an Anthropic or OpenAI model a code sandbox and asks it to find stable marriage instances with as many stable matchings as possible, then verifies and logs the result.

## Setup

Create a `.env` in the repo root:

```bash
cp .env.example .env
```

Add your API keys in the `.env` file.

## Run

```
uv run math_bench.py
```

Set `MATH_MODEL` to `gpt-6.2-sol` or `gpt-6.2-astra` with `MATH_PROVIDER=openai`. Both use the existing Responses API runner.

Results go to `out/math/<model>/<datetime>_<model>/` as `run.json` and `run.md`.

Tests: `uv run python -m unittest`

## Roadmap

- Design Benchmark
- Theology Benchmark
