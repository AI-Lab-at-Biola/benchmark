"""Math track: Maximizing stable matchings (docs/math/starter_doc.md).

Run once a day:  uv run math_bench.py
Config comes from .env; MATH_PROVIDER picks Anthropic or OpenAI.
Writes ./math/<datetime>_<model>/run.json and run.md.
"""

import hashlib
import itertools
import json
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
)

PROMPT_VERSION = "math-v2"
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "out/math"
WALL_CLOCK_CAP_MIN = 15

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


def load_config():
    load_dotenv(HERE / ".env")
    provider = os.environ.get("MATH_PROVIDER", "").strip()
    if provider not in API_KEY_VARS:
        sys.exit(f"MATH_PROVIDER in .env must be one of: {', '.join(API_KEY_VARS)}")
    required = [
        API_KEY_VARS[provider],
        "MATH_MODEL",
        "MATH_EFFORT",
        "MATH_ORDER",
        "MATH_TARGET",
        "MATH_PREVIOUS_BEST",
        "MATH_WALL_CLOCK_MIN",
        "MATH_MAX_OUTPUT_TOKENS",
        "MATH_MAX_TOOL_CALLS",
    ]
    missing = [key for key in required if not os.environ.get(key, "").strip()]
    if missing:
        sys.exit(f"Missing from .env: {', '.join(missing)}")
    return {
        "provider": provider,
        "model": os.environ["MATH_MODEL"].strip(),
        "effort": os.environ["MATH_EFFORT"].strip(),
        "order": int(os.environ["MATH_ORDER"]),
        "target": int(os.environ["MATH_TARGET"]),
        "previous_best": int(os.environ["MATH_PREVIOUS_BEST"]),
        "wall_clock_min": min(
            float(os.environ["MATH_WALL_CLOCK_MIN"]), WALL_CLOCK_CAP_MIN
        ),
        "max_output_tokens": int(os.environ["MATH_MAX_OUTPUT_TOKENS"]),
        "max_tool_calls": int(os.environ["MATH_MAX_TOOL_CALLS"]),
    }


# --- Prompt (Section 5) ------------------------------------------------------------------------


def build_prompt(config):
    n = config["order"]
    return f"""
    You are working on an open research problem in combinatorics. Nobody knows the answer.

## Problem

An order-{n} stable marriage instance has {n} men and {n} women (both indexed 0..{n - 1}).

- Men's matrix M ({n}x{n}): M[i][j] is the rank man i gives woman j. Each row is a permutation of 1..{n}. Rank 1 is most preferred.
- Women's matrix W ({n}x{n}): W[i][j] is the rank woman j gives man i. Each column is a permutation of 1..{n}. Rank 1 is most preferred.
- Example (n=3): men row [2, 1, 3] means man 0 likes woman 1 best, then woman 0, then woman 2.
- A perfect matching is stable if there is no man i and woman j, not matched to each other, who both rank each other better than their assigned partners.

Objective: find (M, W) that maximizes the number of stable matchings.

## Context

For order {n}, the best known value is {config["target"]}. The long-standing bound before that was {config["previous_best"]}. Matching {config["target"]} is excellent; beating it would be a new result. The true maximum is unknown.

## Environment

You have a code execution sandbox (Python; 1 CPU). There is no internet access and you cannot install packages. Write your own code for everything, including counting stable matchings.

Budget for this run: about {config["wall_clock_min"]:.0f} minutes wall-clock, {config["max_tool_calls"]} tool calls, and {config["max_output_tokens"]:,} output tokens in total. The run is cut off when any of these runs out, so leave room to report your answer.

## What to do

1. Before running anything, explain your strategy and why you expect it to work. 
	1. Do *NOT* use simulated annealing as your method of choice
2. Verify every candidate you report by counting its stable matchings with code.
3. Finish with exactly one fenced JSON block in this format, using the conventions above:

```json
{{
  "order": {n},
  "men": [[...{n} ints...], ...{n} rows...],
  "women": [[...{n} ints...], ...{n} rows...],
  "claimed_stable_matchings": <int>,
  "strategy_summary": "2-5 sentences"
}}
```

4. State your confidence in the count. Do not claim your answer is optimal unless you have a proof.
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
        escalate=phi > config["target"],
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
            f"input {usage['input']:,} · output {usage['output']:,} "
            f"(thinking {usage['thinking']:,}) · cache read {usage['cache_read']:,} "
            f"· cache write {usage['cache_write']:,}",
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
    """./math/<datetime>_<model>/, one folder per run."""
    path = OUTPUT_DIR / f"{started:%Y-%m-%d_%H%M%S}_{model}"
    path.mkdir(parents=True)
    return path


def main():
    config = load_config()
    prompt = build_prompt(config)
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
    print(f"verified Φ: {phi} · claimed: {result['claimed_phi']} · {out}")
    if result["escalate"]:
        print(f"*** Verified Φ above {config['target']}. Escalate to Dr. Marks. ***")


if __name__ == "__main__":
    main()
