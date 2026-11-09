## Problem

Find an order-9 stable marriage instance with as many stable matchings as you can.
Both sides are indexed 0..8.
- M[i][j] is man i's rank of woman j. Each of its 9 rows is a permutation of 1..9.
- W[i][j] is woman j's rank of man i. Each of its 9 columns is a permutation of 1..9.
- Rank 1 is most preferred. A perfect matching is stable if no unmatched pair prefers each other to their assigned partners.

No comparison target is supplied; report the best count you verify.

## Environment and budget

Use the provided Python code sandbox. Do not use internet access or install packages.
Write your own code, including the stable-matching counter.
The run budget is 15 minutes, 30 tool calls,
and 16,000 output tokens in total. Reserve time and tokens for verification and your final answer.
Keep the strategy explanation, progress text, and tool output brief. Do not print long search logs or repeat candidate matrices.

## Instructions

1. Briefly state your strategy before executing code. Do NOT use simulated annealing as your method of choice.
2. Count stable matchings with code for every candidate you report. Do not claim optimality without a proof.
3. Finish with exactly one fenced JSON block below, replacing placeholders with actual integer matrices and the verified count. Include count confidence in the short strategy summary.

```json
{
  "order": 9,
  "men": [[...9 ints...], ...9 rows...],
  "women": [[...9 ints...], ...9 rows...],
  "claimed_stable_matchings": <int>,
  "strategy_summary": "Brief strategy and confidence in the count"
}
```
