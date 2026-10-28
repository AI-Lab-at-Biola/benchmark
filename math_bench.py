"""Math track: Maximizing stable matchings (docs/math/starter_doc.md).

Run once a day:  uv run math_bench.py
Config comes from .env. Writes ./math/<date>_<model>.pdf and a matching .json.
"""

import hashlib
import itertools
import json
import os
import re
import sys
import time
from datetime import date, datetime
from pathlib import Path
from xml.sax.saxutils import escape

import anthropic
from dotenv import load_dotenv
from reportlab import platypus
from reportlab.lib import colors, pagesizes, styles, units
from reportlab.pdfbase import pdfmetrics, ttfonts

PROMPT_VERSION = "math-v1"
CODE_EXECUTION_TOOL = {"type": "code_execution_20260521", "name": "code_execution"}
HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "math"

# Section 10: phrases that suggest the model recalled the paper.
CONTAMINATION_PATTERNS = [r"cross-diagonal", r"truncated Latin square", r"\bMarks\b", r"\bBuzi\b", r"\bEilers\b", r"problem reduction"]

# Section 8.1 ladder, only meaningful for order 7: (lowest phi in tier, tier).
ORDER_7_TIERS = [(82, 5), (81, 4), (71, 3), (39, 2), (11, 1), (0, 0)]


# --- Config -------------------------------------------------------------------------------------

def load_config():
    load_dotenv(HERE / ".env")
    required = [
        "ANTHROPIC_API_KEY", "MATH_MODEL", "MATH_EFFORT", "MATH_ORDER", "MATH_TARGET",
        "MATH_PREVIOUS_BEST", "MATH_WALL_CLOCK_MIN", "MATH_MAX_OUTPUT_TOKENS", "MATH_MAX_TOOL_CALLS",
    ]
    missing = [key for key in required if not os.environ.get(key, "").strip()]
    if missing:
        sys.exit(f"Missing from .env: {', '.join(missing)}")
    return {
        "model": os.environ["MATH_MODEL"].strip(),
        "effort": os.environ["MATH_EFFORT"].strip(),
        "order": int(os.environ["MATH_ORDER"]),
        "target": int(os.environ["MATH_TARGET"]),
        "previous_best": int(os.environ["MATH_PREVIOUS_BEST"]),
        "wall_clock_min": float(os.environ["MATH_WALL_CLOCK_MIN"]),
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


# --- Run the model ----------------------------------------------------------------------------

def run_model(config, prompt):
    """Drive the conversation until the model finishes or a budget runs out. Returns the run record."""
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": prompt}]
    usage = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0, "thinking": 0}
    responses = []  # one entry per API request, in order
    tool_calls = 0
    container_id = None
    stop = {"reason": None, "budget_exhausted": None, "error": None}
    served_model = None
    started = time.monotonic()

    while True:
        elapsed_min = (time.monotonic() - started) / 60
        remaining_tokens = config["max_output_tokens"] - usage["output"]
        if elapsed_min >= config["wall_clock_min"]:
            stop["budget_exhausted"] = "wall_clock"
            break
        if tool_calls >= config["max_tool_calls"]:
            stop["budget_exhausted"] = "tool_calls"
            break
        if remaining_tokens < 1024:
            stop["budget_exhausted"] = "output_tokens"
            break

        request = dict(
            model=config["model"],
            max_tokens=min(64000, remaining_tokens),
            thinking={"type": "adaptive", "display": "summarized"},
            output_config={"effort": config["effort"]},
            tools=[CODE_EXECUTION_TOOL],
            messages=messages,
        )
        if container_id:
            request["container"] = container_id

        try:
            with client.messages.stream(**request) as stream:
                response = stream.get_final_message()
        except anthropic.APIError as error:
            stop["error"] = f"{type(error).__name__}: {error}"
            break

        served_model = response.model
        if response.container:
            container_id = response.container.id
        add_usage(usage, response.usage)
        tool_calls += sum(1 for block in response.content if block.type == "server_tool_use")
        responses.append({"stop_reason": response.stop_reason, "blocks": [block_record(b) for b in response.content]})
        messages.append({"role": "assistant", "content": response.content})
        stop["reason"] = response.stop_reason

        if response.stop_reason == "pause_turn":
            # The server's tool loop hit its iteration limit; resending the history resumes it.
            continue
        if response.stop_reason == "max_tokens":
            # Prefill isn't allowed, so a cut-off turn continues through a user message.
            messages.append({"role": "user", "content": "You hit the per-response output limit. Continue where you left off."})
            continue
        break  # end_turn, refusal, or anything unexpected

    return {
        "served_model": served_model,
        "usage": usage,
        "tool_calls": tool_calls,
        "wall_clock_min": round((time.monotonic() - started) / 60, 2),
        "stop": stop,
        "responses": responses,
    }


def add_usage(totals, usage):
    totals["input"] += usage.input_tokens or 0
    totals["output"] += usage.output_tokens or 0
    totals["cache_read"] += usage.cache_read_input_tokens or 0
    totals["cache_write"] += usage.cache_creation_input_tokens or 0
    details = getattr(usage, "output_tokens_details", None)
    totals["thinking"] += getattr(details, "thinking_tokens", 0) or 0


def block_record(block):
    """Reduce an API content block to the plain dict we log and render."""
    if block.type == "thinking":
        return {"type": "thinking", "text": block.thinking}
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "server_tool_use":
        return {"type": "tool_call", "tool": block.name, "input": block.input}
    if block.type == "bash_code_execution_tool_result":
        result = block.content
        if result.type == "bash_code_execution_result":
            return {"type": "tool_output", "return_code": result.return_code, "stdout": result.stdout, "stderr": result.stderr}
        return {"type": "tool_output", "error": getattr(result, "error_code", str(result))}
    if block.type == "text_editor_code_execution_tool_result":
        return {"type": "tool_output", "editor": block.content.to_dict()}
    return {"type": block.type, "raw": block.to_dict()}


# --- Verification (Section 7) -----------------------------------------------------------------

def extract_answer(run):
    """The fenced JSON block from the model's final response, or None."""
    if not run["responses"]:
        return None
    final_text = "\n".join(b["text"] for b in run["responses"][-1]["blocks"] if b["type"] == "text")
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
        if not (isinstance(matrix, list) and len(matrix) == n and all(isinstance(r, list) and len(r) == n for r in matrix)):
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
            for i in range(n) for j in range(n)
        ):
            count += 1
    return count


def tier_for(phi, n):
    if n != 7 or phi is None:
        return None
    return next(tier for lowest, tier in ORDER_7_TIERS if phi >= lowest)


def contamination_hits(run):
    text = json.dumps(run["responses"])
    return sorted({p for p in CONTAMINATION_PATTERNS if re.search(p, text, re.IGNORECASE if p.islower() else 0)})


def score(config, run):
    answer = extract_answer(run)
    result = {
        "valid": False, "invalid_reason": None, "verified_phi": None, "claimed_phi": None,
        "tier": None, "honesty_flag": False, "escalate": False, "answer": answer,
        "contamination_hits": contamination_hits(run),
    }
    if answer is None:
        result["invalid_reason"] = "format failure: no parseable JSON block in the final response"
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

def register_fonts():
    """Arial and Menlo cover symbols like Φ and ≤ that the built-in PDF fonts can't draw."""
    fonts = {"body": "Helvetica", "bold": "Helvetica-Bold", "italic": "Helvetica-Oblique", "mono": "Courier"}
    supplemental = Path("/System/Library/Fonts/Supplemental")
    try:
        pdfmetrics.registerFont(ttfonts.TTFont("Arial", supplemental / "Arial.ttf"))
        pdfmetrics.registerFont(ttfonts.TTFont("Arial-Bold", supplemental / "Arial Bold.ttf"))
        pdfmetrics.registerFont(ttfonts.TTFont("Arial-Italic", supplemental / "Arial Italic.ttf"))
        fonts.update(body="Arial", bold="Arial-Bold", italic="Arial-Italic")
    except Exception:
        pass
    try:
        pdfmetrics.registerFont(ttfonts.TTFont("Menlo", "/System/Library/Fonts/Menlo.ttc", subfontIndex=0))
        fonts["mono"] = "Menlo"
    except Exception:
        pass
    return fonts


def write_pdf(path, config, prompt, run, result):
    fonts = register_fonts()
    base = styles.getSampleStyleSheet()
    body = styles.ParagraphStyle("body", parent=base["BodyText"], fontName=fonts["body"], fontSize=9.5, leading=13)
    para_styles = {
        "title": styles.ParagraphStyle("title", parent=base["Title"], fontName=fonts["bold"], fontSize=16),
        "h2": styles.ParagraphStyle("h2", parent=base["Heading2"], fontName=fonts["bold"], fontSize=12, spaceBefore=10),
        "label": styles.ParagraphStyle("label", parent=body, fontName=fonts["bold"], spaceBefore=6),
        "body": body,
        "thinking": styles.ParagraphStyle("thinking", parent=body, fontName=fonts["italic"], textColor=colors.HexColor("#555555"), leftIndent=8),
        "mono": styles.ParagraphStyle("mono", parent=body, fontName=fonts["mono"], fontSize=7.5, leading=9.5,
                               backColor=colors.HexColor("#f4f4f4"), borderPadding=4, leftIndent=4, rightIndent=4),
    }

    def para(text, style="body"):
        return platypus.Paragraph(escape(text).replace("\n", "<br/>"), para_styles[style])

    def code(text, max_lines=300):
        lines = text.splitlines() or [""]
        if len(lines) > max_lines:
            lines = lines[:max_lines] + [f"... ({len(lines) - max_lines} more lines in the .json)"]
        return platypus.Preformatted("\n".join(lines), para_styles["mono"], maxLineLength=110, newLineChars="")

    def table(rows, widths=None):
        t = platypus.Table([[para(str(c)) for c in row] for row in rows], colWidths=widths, hAlign="LEFT")
        t.setStyle(platypus.TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.grey), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    usage, stop = run["usage"], run["stop"]
    story = [
        para(f"Math track: {config['model']} · {date.today().isoformat()}", "title"),
        para("Summary", "h2"),
        table([
            ["Verified Φ", result["verified_phi"] if result["valid"] else f"INVALID: {result['invalid_reason']}"],
            ["Claimed Φ", result["claimed_phi"]],
            ["Tier (n=7 ladder)", result["tier"]],
            ["Flags", ", ".join(f for f, on in [
                ("claimed ≠ verified", result["honesty_flag"]),
                (f"Φ > {config['target']}: ESCALATE to Dr. Marks", result["escalate"]),
                (f"contamination terms: {', '.join(result['contamination_hits'])}", result["contamination_hits"]),
                (f"budget exhausted ({stop['budget_exhausted']})", stop["budget_exhausted"]),
                (f"API error: {stop['error']}", stop["error"]),
            ] if on) or "none"],
            ["Stop reason", stop["reason"]],
            ["Model (requested / served)", f"{config['model']} / {run['served_model']}"],
            ["Effort · prompt version", f"{config['effort']} · {PROMPT_VERSION}"],
            ["Tokens", f"input {usage['input']:,} · output {usage['output']:,} (thinking {usage['thinking']:,}) · "
                       f"cache read {usage['cache_read']:,} · cache write {usage['cache_write']:,}"],
            ["Tool calls · wall-clock", f"{run['tool_calls']} · {run['wall_clock_min']} min"],
            ["Budget", f"{config['wall_clock_min']:.0f} min · {config['max_tool_calls']} tool calls · {config['max_output_tokens']:,} output tokens"],
        ], widths=[1.8 * units.inch, 5.2 * units.inch]),
    ]

    answer = result["answer"]
    if answer:
        story.append(para("Submitted profile", "h2"))
        story.append(para(answer.get("strategy_summary", "(no strategy_summary)")))
        for name in ("men", "women"):
            matrix = answer.get(name)
            if isinstance(matrix, list):
                story.append(para(f"{name}", "label"))
                story.append(code("\n".join(" ".join(f"{v:>2}" if isinstance(v, int) else str(v) for v in row) if isinstance(row, list) else str(row) for row in matrix)))

    story += [platypus.PageBreak(), para("Transcript", "h2"),
              para("Thinking is Anthropic's summarized reasoning, not the full chain of thought.", "thinking")]
    for number, response in enumerate(run["responses"], start=1):
        story.append(para(f"Response {number} (stop: {response['stop_reason']})", "label"))
        for block in response["blocks"]:
            kind = block["type"]
            if kind == "thinking" and block["text"]:
                story += [para("Thinking", "label"), para(block["text"], "thinking")]
            elif kind == "text":
                story += [para("Model", "label"), para(block["text"])]
            elif kind == "tool_call":
                inp = block["input"]
                text = inp.get("command") if block["tool"] == "bash_code_execution" else json.dumps(inp, indent=2)
                if block["tool"] == "text_editor_code_execution" and "file_text" in inp:
                    text = f"# {inp.get('command')} {inp.get('path')}\n{inp['file_text']}"
                story += [para(f"Code ({block['tool']})", "label"), code(str(text))]
            elif kind == "tool_output":
                if "stdout" in block:
                    text = block["stdout"] + (f"\n[stderr]\n{block['stderr']}" if block["stderr"] else "")
                    story += [para(f"Output (exit {block['return_code']})", "label"), code(text)]
                else:
                    story += [para("Output", "label"), code(json.dumps(block.get("editor") or block.get("error"), indent=2))]
        story.append(platypus.Spacer(1, 6))

    story += [platypus.PageBreak(), para("Appendix: prompt", "h2"), code(prompt, max_lines=1000)]
    platypus.SimpleDocTemplate(str(path), pagesize=pagesizes.letter, title=f"Math track {config['model']}",
                      leftMargin=0.7 * units.inch, rightMargin=0.7 * units.inch, topMargin=0.7 * units.inch, bottomMargin=0.7 * units.inch).build(story)


def output_stem(model):
    """./math/<date>_<model>, with a numeric suffix if today's run already exists."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    stem = OUTPUT_DIR / f"{date.today().isoformat()}_{model}"
    suffix = 2
    candidate = stem
    while candidate.with_suffix(".pdf").exists():
        candidate = stem.with_name(f"{stem.name}_{suffix}")
        suffix += 1
    return candidate


def main():
    config = load_config()
    prompt = build_prompt(config)
    started_at = datetime.now().astimezone().isoformat(timespec="seconds")
    run = run_model(config, prompt)
    result = score(config, run)

    stem = output_stem(config["model"])
    record = {
        "started_at": started_at,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "code_execution_tool": CODE_EXECUTION_TOOL["type"],
        "config": config,
        "result": result,
        "run": run,
    }
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2, default=str))
    write_pdf(stem.with_suffix(".pdf"), config, prompt, run, result)

    phi = result["verified_phi"] if result["valid"] else f"invalid ({result['invalid_reason']})"
    print(f"verified Φ: {phi} · claimed: {result['claimed_phi']} · {stem.with_suffix('.pdf')}")
    if result["escalate"]:
        print(f"*** Verified Φ above {config['target']}. Escalate to Dr. Marks. ***")


if __name__ == "__main__":
    main()
