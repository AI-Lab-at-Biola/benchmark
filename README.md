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

Results go to `out/math/<datetime>_<model>/` as `run.json` and `run.md`.

Tests: `uv run python -m unittest test_model_calls.py`

## Roadmap

- Design Benchmark
- Theology Benchmark
