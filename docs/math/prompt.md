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