"""Math track: Maximizing stable matchings (docs/math/math_v4.md).

Run manually: uv run math_bench.py (preview with --dry-run)
Config comes from .env; MATH_PROVIDER picks Anthropic or OpenAI.
Writes ./out/math/<datetime>_<model>/run.json and run.md.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from model_calls import (
    API_KEY_VARS,
    CODE_EXECUTION_TOOL,
    OPENAI_CODE_TOOL,
    run_model,
    run_openai,
    usage_status,
)

PROMPT_VERSION = "math-v4"
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "out/math"

# Section 10: phrases that suggest the model recalled the paper.
CONTAMINATION_PATTERNS = [
    r"cross-diagonal",
    r"truncated Latin square",
    r"\bMarks\b",
    r"\bBuzi\b",
    r"\bEilers\b",
    r"problem reduction",
]

# Section 8.1 ladder, only meaningful for order 7: (lowest phi in tier, tier).
ORDER_7_TIERS = [(82, 5), (81, 4), (71, 3), (39, 2), (11, 1), (0, 0)]


# --- Config -------------------------------------------------------------------------------------


def load_config(*, dry_run=False):
    load_dotenv(HERE / ".env")
    provider = os.environ.get("MATH_PROVIDER", "").strip()
    if provider not in API_KEY_VARS:
        sys.exit(f"MATH_PROVIDER in .env must be one of: {', '.join(API_KEY_VARS)}")
    required = [
        "MATH_MODEL",
        "MATH_EFFORT",
        "MATH_ORDER",
        "MATH_WALL_CLOCK_MIN",
        "MATH_MAX_OUTPUT_TOKENS",
        "MATH_MAX_TOOL_CALLS",
    ]
    if not dry_run:
        required.append(API_KEY_VARS[provider])
    missing = [key for key in required if not os.environ.get(key, "").strip()]
    if missing:
        sys.exit(f"Missing from .env: {', '.join(missing)}")
    model = os.environ["MATH_MODEL"].strip()
    try:
        config = {
            "provider": provider,
            "model": model,
            "effort": os.environ["MATH_EFFORT"].strip(),
            "order": int(os.environ["MATH_ORDER"]),
            "target": optional_int("MATH_TARGET"),
            "previous_best": optional_int("MATH_PREVIOUS_BEST"),
            "wall_clock_min": float(os.environ["MATH_WALL_CLOCK_MIN"]),
            "max_output_tokens": int(os.environ["MATH_MAX_OUTPUT_TOKENS"]),
            "max_tool_calls": int(os.environ["MATH_MAX_TOOL_CALLS"]),
        }
    except ValueError as exc:
        sys.exit(f"Invalid numeric configuration: {exc}")
    for key in ("order", "wall_clock_min", "max_output_tokens", "max_tool_calls"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            sys.exit(f"{key} must be positive and finite")
    if config["max_output_tokens"] < 1024:
        sys.exit("max_output_tokens must be at least 1024")
    for key in ("target", "previous_best"):
        if config[key] is not None and config[key] < 0:
            sys.exit(f"{key} must be nonnegative when configured")
    return config


def optional_int(name):
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


# --- Prompt ----------------------------------------------------------------------------------


def build_prompt(config):
    n = config["order"]
    comparisons = []
    if config.get("target") is not None:
        comparisons.append(f"Configured comparison target: {config['target']}.")
    if config.get("previous_best") is not None:
        comparisons.append(
            f"Configured previous-run comparison: {config['previous_best']}."
        )
    context = " ".join(comparisons)
    if context:
        context += (
            " These are supplied comparison values, not verified research records."
        )
    else:
        context = "No comparison target is supplied; report the best count you verify."
    return f"""## Problem

Find an order-{n} stable marriage instance with as many stable matchings as you can.
Both sides are indexed 0..{n - 1}.
- M[i][j] is man i's rank of woman j. Each of its {n} rows is a permutation of 1..{n}.
- W[i][j] is woman j's rank of man i. Each of its {n} columns is a permutation of 1..{n}.
- Rank 1 is most preferred. A perfect matching is stable if no unmatched pair prefers each other to their assigned partners.

{context}

## Environment and budget

Use the provided Python code sandbox. Do not use internet access or install packages.
Write your own code, including the stable-matching counter.
The run budget is {config["wall_clock_min"]:g} minutes, {config["max_tool_calls"]} tool calls,
and {config["max_output_tokens"]:,} output tokens in total. Reserve time and tokens for verification and your final answer.
Keep the strategy explanation, progress text, and tool output brief. Do not print long search logs or repeat candidate matrices.

## Instructions

1. Briefly state your strategy before executing code. Do NOT use simulated annealing as your method of choice.
2. Count stable matchings with code for every candidate you report. Do not claim optimality without a proof.
3. Finish with exactly one fenced JSON block below, replacing placeholders with actual integer matrices and the verified count. Include count confidence in the short strategy summary.

```json
{{
  "order": {n},
  "men": [[...{n} ints...], ...{n} rows...],
  "women": [[...{n} ints...], ...{n} rows...],
  "claimed_stable_matchings": <int>,
  "strategy_summary": "Brief strategy and confidence in the count"
}}
```
"""


# --- Verification (Section 7) -----------------------------------------------------------------


def extract_answer(run):
    """The fenced JSON block from the model's final response, or None."""
    if not run["responses"]:
        return None
    final_text = "\n".join(
        b["text"] for b in run["responses"][-1]["blocks"] if b["type"] == "text"
    )
    blocks = re.findall(r"```json\s*(.*?)```", final_text, re.DOTALL)
    if not blocks:
        return None
    try:
        return json.loads(blocks[-1])
    except json.JSONDecodeError:
        return None


def validate_profile(answer, n):
    """Problems with the profile's structure; empty means valid."""
    problems = []
    men, women = answer.get("men"), answer.get("women")
    if answer.get("order") != n:
        problems.append(f"order is {answer.get('order')}, expected {n}")
    for name, matrix in (("men", men), ("women", women)):
        if not (
            isinstance(matrix, list)
            and len(matrix) == n
            and all(isinstance(r, list) and len(r) == n for r in matrix)
        ):
            problems.append(f"{name} is not a {n}x{n} list of lists")
    if problems:
        return problems
    expected = list(range(1, n + 1))
    for i, row in enumerate(men):
        if sorted(row) != expected:
            problems.append(f"men row {i} is not a permutation of 1..{n}")
    for j in range(n):
        if sorted(women[i][j] for i in range(n)) != expected:
            problems.append(f"women column {j} is not a permutation of 1..{n}")
    return problems


def count_stable_matchings(men, women):
    """Brute force: try every perfect matching and check it for blocking pairs.

    men[i][j] = man i's rank of woman j; women[i][j] = woman j's rank of man i; lower is better.
    """
    n = len(men)
    count = 0
    for wife in itertools.permutations(range(n)):  # wife[i] = woman matched to man i
        husband = [0] * n
        for man, woman in enumerate(wife):
            husband[woman] = man
        if not any(
            men[i][j] < men[i][wife[i]] and women[i][j] < women[husband[j]][j]
            for i in range(n)
            for j in range(n)
        ):
            count += 1
    return count


def tier_for(phi, n):
    if n != 7 or phi is None:
        return None
    return next(tier for lowest, tier in ORDER_7_TIERS if phi >= lowest)


def contamination_hits(run):
    text = json.dumps(run["responses"])
    return sorted(
        {
            p
            for p in CONTAMINATION_PATTERNS
            if re.search(p, text, re.IGNORECASE if p.islower() else 0)
        }
    )


def score(config, run):
    answer = extract_answer(run)
    result = {
        "valid": False,
        "invalid_reason": None,
        "verified_phi": None,
        "claimed_phi": None,
        "tier": None,
        "honesty_flag": False,
        "escalate": False,
        "answer": answer,
        "contamination_hits": contamination_hits(run),
    }
    if answer is None:
        result["invalid_reason"] = (
            "format failure: no parseable JSON block in the final response"
        )
        return result
    result["claimed_phi"] = answer.get("claimed_stable_matchings")
    problems = validate_profile(answer, config["order"])
    if problems:
        result["invalid_reason"] = "; ".join(problems)
        return result
    phi = count_stable_matchings(answer["men"], answer["women"])
    result.update(
        valid=True,
        verified_phi=phi,
        tier=tier_for(phi, config["order"]),
        honesty_flag=result["claimed_phi"] != phi,
        escalate=config.get("target") is not None and phi > config["target"],
    )
    return result


# --- Report ----------------------------------------------------------------------------------


def fenced(text, lang=""):
    """A Markdown code block whose fence can't collide with backticks inside the text."""
    longest = max((len(ticks) for ticks in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{text.rstrip()}\n{fence}"


def truncated(text, max_lines=300):
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    hidden = len(lines) - max_lines
    return "\n".join(lines[:max_lines] + [f"... ({hidden} more lines in run.json)"])


def flag_summary(config, run, result):
    stop = run["stop"]
    hits = ", ".join(result["contamination_hits"])
    checks = [
        ("claimed ≠ verified", result["honesty_flag"]),
        (f"Φ > {config['target']}: ESCALATE to Dr. Marks", result["escalate"]),
        (f"contamination terms: {hits}", result["contamination_hits"]),
        (f"budget exhausted ({stop['budget_exhausted']})", stop["budget_exhausted"]),
        (f"API error: {stop['error']}", stop["error"]),
    ]
    return ", ".join(label for label, on in checks if on) or "none"


def block_markdown(block):
    """Markdown lines for one transcript block; empty for blocks with nothing to show."""
    kind = block["type"]
    if kind == "thinking":
        if not block["text"]:
            return []
        quoted = "\n".join(f"> {line}" for line in block["text"].splitlines())
        return ["**Thinking**", "", quoted]
    if kind == "text":
        return ["**Model**", "", block["text"]]
    if kind == "tool_call":
        inp = block["input"]
        if block["tool"] == "bash_code_execution":
            return ["**Code** (bash)", "", fenced(str(inp.get("command")), "bash")]
        if block["tool"] == "code_interpreter":
            return ["**Code** (python)", "", fenced(str(inp.get("code")), "python")]
        if "file_text" in inp:
            path = str(inp.get("path", ""))
            lang = "python" if path.endswith(".py") else ""
            label = f"**Code** ({inp.get('command')} `{path}`)"
            return [label, "", fenced(inp["file_text"], lang)]
        return ["**Code**", "", fenced(json.dumps(inp, indent=2), "json")]
    if kind == "tool_output":
        if "stdout" in block:
            text = block["stdout"]
            if block["stderr"]:
                text += f"\n[stderr]\n{block['stderr']}"
            if block["return_code"] is None:
                label = f"**Output** ({block.get('status')})"
            else:
                label = f"**Output** (exit {block['return_code']})"
            return [label, "", fenced(truncated(text))]
        detail = block.get("editor") or block.get("error")
        return ["**Output**", "", fenced(json.dumps(detail, indent=2), "json")]
    return [f"**{kind}**", "", fenced(json.dumps(block, indent=2, default=str), "json")]


def write_markdown(path, config, prompt, run, result, started):
    usage = run["usage"]
    verified = (
        result["verified_phi"]
        if result["valid"]
        else f"INVALID: {result['invalid_reason']}"
    )
    summary = [
        ("Verified Φ", verified),
        ("Claimed Φ", result["claimed_phi"]),
        ("Tier (n=7 ladder)", result["tier"]),
        ("Flags", flag_summary(config, run, result)),
        ("Stop reason", run["stop"]["reason"]),
        (
            "Model (requested / served)",
            f"{config['provider']}: {config['model']} / {run['served_model']}",
        ),
        ("Effort · prompt version", f"{config['effort']} · {PROMPT_VERSION}"),
        (
            "Tokens",
            usage_status(
                usage, config["provider"], complete=run.get("usage_complete", True)
            ),
        ),
        (
            "Tool calls · wall-clock",
            f"{run['tool_calls']} · {run['wall_clock_min']} min",
        ),
        (
            "Budget",
            f"{config['wall_clock_min']:.0f} min · {config['max_tool_calls']} tool calls "
            f"· {config['max_output_tokens']:,} output tokens",
        ),
    ]
    lines = [
        f"# Math track: {config['model']} · {started:%Y-%m-%d %H:%M}",
        "",
        "## Summary",
        "",
        "| | |",
        "| --- | --- |",
    ]
    for label, value in summary:
        cell = str(value).replace("|", "\\|")
        lines.append(f"| {label} | {cell} |")

    answer = result["answer"]
    if answer:
        strategy = answer.get("strategy_summary", "(no strategy_summary)")
        lines += ["", "## Submitted profile", "", str(strategy)]
        for name in ("men", "women"):
            matrix = answer.get(name)
            rows = matrix if isinstance(matrix, list) else [matrix]
            lines += [
                "",
                f"**{name}**",
                "",
                fenced("\n".join(json.dumps(r) for r in rows)),
            ]

    lines += [
        "",
        "## Transcript",
        "",
        "_Thinking is the provider's summarized reasoning, not the full chain of thought._",
    ]
    for number, response in enumerate(run["responses"], start=1):
        lines += ["", f"### Response {number} (stop: {response['stop_reason']})"]
        for block in response["blocks"]:
            block_lines = block_markdown(block)
            if block_lines:
                lines += ["", *block_lines]

    lines += ["", "## Appendix: prompt", "", fenced(prompt, "markdown")]
    path.write_text("\n".join(lines) + "\n")


def run_dir(started, model):
    """./out/math/<datetime>_<model>/, one folder per run."""
    path = OUTPUT_DIR / f"{started:%Y-%m-%d_%H%M%S}_{model}"
    path.mkdir(parents=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview configuration and prompt without API calls or output files",
    )
    args = parser.parse_args(argv)
    config = load_config(dry_run=args.dry_run)
    prompt = build_prompt(config)
    if args.dry_run:
        print(
            json.dumps({"prompt_version": PROMPT_VERSION, "config": config}, indent=2)
        )
        print("\n" + prompt)
        return
    started = datetime.now().astimezone()
    is_openai = config["provider"] == "openai"
    run = run_openai(config, prompt) if is_openai else run_model(config, prompt)
    result = score(config, run)

    out = run_dir(started, config["model"])
    record = {
        "started_at": started.isoformat(timespec="seconds"),
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "code_execution_tool": OPENAI_CODE_TOOL if is_openai else CODE_EXECUTION_TOOL,
        "config": config,
        "result": result,
        "run": run,
    }
    (out / "run.json").write_text(json.dumps(record, indent=2, default=str))
    write_markdown(out / "run.md", config, prompt, run, result, started)

    phi = (
        result["verified_phi"]
        if result["valid"]
        else f"invalid ({result['invalid_reason']})"
    )
    print(
        usage_status(
            run["usage"], config["provider"], complete=run.get("usage_complete", True)
        )
    )
    print(f"verified Φ: {phi} · claimed: {result['claimed_phi']} · {out}")
    if result["escalate"]:
        print(f"*** Verified Φ above {config['target']}. Escalate to Dr. Marks. ***")


if __name__ == "__main__":
    main()
