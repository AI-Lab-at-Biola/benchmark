# Biola AI Lab Benchmark — Math Track Rules

## Maximizing Stable Matchings in the Stable Marriage Problem

**Status:** Draft v0.1 · **Owner:** Ben Currie · **Advisor input:** Dr. Justin Marks
**Basis:** Marks, Buzi & Eilers, _Improved Hill Climbing Search by Heuristic Seed Selection via Problem Reduction_

---

## 1. Purpose

This track measures how a model reasons on a **genuinely open, objectively verifiable** math problem. It produces two independent signals:

1. **Objective score.** The best verified number of stable matchings the model finds.
2. **Creativity score.** How novel and structurally insightful the model's approach is, graded by a human against a fixed rubric.

These feed the lab benchmark's split: roughly **70% numbers** (tokens, score, tool usage) and **30% human-reviewed output quality**.

---

## 2. Why This Problem

- **It's open.** Finding the maximum number of stable matchings for order _n_ is Knuth's Research Problem #5 (1976). The exact maximum is known only up to _n_ = 5 (value 16).
- **Every answer can be verified.** Any proposed preference profile can be counted mechanically. The model cannot "sound right"; its answer is either verified or it isn't.
- **There is a fresh record to beat.** The best known value for _n_ = 7 is **81** (Marks et al.), up from Thurber's long-standing **71**. Public sources such as OEIS A357271 still list 71, so current models likely don't know about 81. That lowers contamination risk.
- **It resists recall.** No known optimal answer exists to memorize.

---

## 3. Problem Statement (given to the model)

- An order-_n_ instance has _n_ men and _n_ women.
- **Men's matrix M (n×n).** Row _i_ is man _i_'s ranking of each woman. Each row is a permutation of 1.._n_.
- **Women's matrix W (n×n).** Column _j_ is woman _j_'s ranking of each man. Each column is a permutation of 1.._n_.
- A matching is **stable** if no man and woman both prefer each other over their assigned partners.
- **Objective:** find (M, W) that maximizes Φ, the number of stable matchings.

### Target orders

| Order | Role                 | Best known       | Notes                                                                     |
| ----- | -------------------- | ---------------- | ------------------------------------------------------------------------- |
| **7** | **Primary**          | **81**           | Fresh record. Odd orders appear to have the most headroom.                |
| 9     | Secondary (optional) | 330 (Thurber LB) | Likely improvable. Use only once the n=7 pipeline is stable.              |
| 8     | **Excluded**         | 268              | Conjectured exact. "Beat it" may be impossible, so there's little signal. |

---

## 4. Environment Rules

| Rule                | Requirement                                                                                                                                                         |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Network             | **None.** The container has no egress, enforced at the container and network level, not by prompt.                                                                  |
| Tools               | Tool calling only: code execution (Python, optionally C++) and a scratch filesystem inside the container.                                                           |
| Packages            | Pre-installed, pinned image. No package installation at runtime (no network anyway). Image hash is logged per run.                                                  |
| Pre-loaded material | **Nothing problem-specific.** No papers, no reference profiles, no counting code. The model writes its own.                                                         |
| Compute budget      | Fixed per run. Proposed defaults, to tune after a pilot: **60 min wall-clock, fixed max output-token cap, fixed max tool calls.** Identical across models and days. |
| Model settings      | Fixed model ID string, fixed effort/thinking settings, summarized thinking enabled. Logged per run.                                                                 |
| API keys            | Separate key per model per track, for token isolation. Keys are never stored in shared docs: local storage or Secret Manager only.                                  |

---

## 5. Prompt Rules

- **One fixed, version-controlled prompt.** Any change bumps the prompt version, and results are only compared within a version.
- The prompt **states the target (81)** and the old bound (71) as context.
- The prompt **does not describe the paper's method** (problem-reduction seeds from a cross-diagonal Latin square, then steepest-ascent hill climbing). Describing it anchors the model and destroys the creativity signal.
- The prompt asks the model to:
  1. Explain its strategy before executing it.
  2. Verify its own candidates with code.
  3. Report its best profile in the required output format (Section 6).
  4. State its confidence and **not** claim optimality without proof.

---

## 6. Required Output Format

The final answer must contain one fenced JSON block:

```json
{
  "order": 7,
  "men": [[...7 ints...], ...7 rows...],
  "women": [[...7 ints...], ...7 rows...],
  "claimed_stable_matchings": 81,
  "strategy_summary": "2–5 sentences"
}
```

- `men[i][j]` is man _i_'s rank of woman _j_.
- `women[i][j]` is woman _j_'s rank of man _i_, using the same layout as the paper's (M, W) pairs.
- A missing or unparseable block makes the run **invalid** (objective score = 0, logged as a format failure).

---

## 7. Verification (non-negotiable)

1. **The model's self-reported count is never trusted.**
2. Validate the structure. Each row of `men` must be a permutation of 1.._n_, and each column of `women` must be a permutation of 1.._n_. If either check fails, the run is invalid.
3. Count Φ with the **lab's own reference counter**, cross-checked once against the paper's C++ code (`github.com/Rrazr/StableMarriageProblem`).
4. Record both `verified_phi` and `claimed_phi`. A mismatch is an **honesty/calibration flag**.
5. Any verified Φ > 81 is escalated immediately to Dr. Marks for independent confirmation.

---

## 8. Scoring

### 8.1 Objective score (automated)

Ladder benchmarked against the paper's order-7 results:

| Tier | Verified Φ     | Meaning                                                  |
| ---- | -------------- | -------------------------------------------------------- |
| 0    | invalid / 1–10 | No better than random profiles (random max observed: 10) |
| 1    | 11–38          | Random-seed hill-climbing territory (max 38)             |
| 2    | 39–70          | Strong, but below Thurber's 2002 bound                   |
| 3    | 71–80          | Matches or beats Thurber                                 |
| 4    | 81             | Ties the current record                                  |
| 5    | > 81           | **New record** (verify and escalate)                     |

Report the raw `verified_phi` as well. Tiers are for dashboards only.

### 8.2 Creativity score (human, weekly)

Score each dimension 0–3, using the summarized reasoning and the code the model wrote.

| Dimension                | 0                                        | 3                                                                                                                                                               |
| ------------------------ | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Novelty vs. baseline** | Plain random search or random restart    | Approach clearly distinct from seed reduction + hill climbing (e.g., composition of smaller instances, constraint/SAT formulation, new neighborhood definition) |
| **Structural insight**   | Treats the problem as a black box        | Exploits real structure (Latin squares, symmetry, rotation posets, recursive substructure)                                                                      |
| **Search discipline**    | Never checks its own work                | Verifies candidates and iterates on evidence                                                                                                                    |
| **Calibration**          | Claims optimality, or claimed ≠ verified | Accurate claimed count; correctly states that optimality is unproven                                                                                            |

**Creativity and objective scores are reported separately and never blended.** A novel approach scoring 60 is informative; a rehash scoring 81 is also informative. They are different signals.

---

## 9. Metrics Logged Per Run

- Model ID, prompt version, container image hash, timestamp
- Input, output, cache-read, cache-write, and reasoning tokens
- Number of tool calls, total wall-clock, whether the budget was exhausted
- `verified_phi`, `claimed_phi`, validity flags
- Summarized thinking block (raw), final response (raw), all code the model wrote
- Contamination flags (Section 10)

---

## 10. Contamination Controls

- No network: enforced by the container, not by trust.
- Grep every output for paper-specific signals: "cross-diagonal", "truncated Latin square", "Marks", "Buzi", "Eilers", "problem reduction" plus "81". A hit is flagged for review. It doesn't automatically disqualify the run, but the creativity score must account for it.
- If a model reproduces the paper's exact Fig. 11 profile, flag it as probable recall.

---

## 11. Cadence & Statistics

- **3 runs per model per day.** One run per day can't separate drift from run-to-run variance.
- Track the daily **median and max** `verified_phi`, plus median tokens.
- Treat a drift signal as real only when it persists across multiple days, not on a single bad run.
- Human creativity review happens weekly, alongside the other tracks.

---

## 12. Known Limitations

- **Reasoning is a summary.** The thinking block is a summarized version of the model's reasoning, not the full chain of thought. Creativity grading uses this proxy plus the model's code. Token counts still reflect the full reasoning.
- **Ceiling effects.** If no model ever passes tier 2, the objective signal is flat. Mitigation: add order 9 or relax the budget once the baseline is known.
- **Budget sensitivity.** Scores scale with compute. Results are only comparable under an identical budget and prompt version.
- **Grader subjectivity.** The creativity rubric reduces subjectivity but doesn't eliminate it. Use two graders where possible and log disagreements.

---

## 13. Open Questions

- [ ] Final compute budget numbers, after a pilot run
- [ ] Whether to add a second prompt variant that _does_ describe the paper's method, as a control for "can it improve a known technique"
- [ ] Whether order 9 is worth the extra compute
- [ ] Review with Dr. Marks: rubric dimensions and escalation process for > 81 results
- [ ] Loop in Johan on methodology

---

## 14. References

- Marks, Buzi, Eilers. _Improved Hill Climbing Search by Heuristic Seed Selection via Problem Reduction._ Order-7 record of 81; supporting code at `github.com/Rrazr/StableMarriageProblem`.
- Thurber (2002), _Concerning the maximum number of stable matchings in the stable marriage problem_, Discrete Math 248.
- OEIS A357269 (exact maxima, known to n=5) and A357271 (Thurber lower bounds).
- Knuth (1976), _Mariages Stables_, Research Problem #5.
