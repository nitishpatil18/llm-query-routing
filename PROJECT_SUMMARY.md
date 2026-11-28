# Project summary: Adaptive Cost-Aware Query Routing for LLMs

Repository: github.com/nitishpatil18/llm-query-routing

## 1. The question

When does it pay to route a query to a cheaper LLM instead of a stronger one, and what signal should drive that decision? The project measures this end to end on real model calls (no simulated or estimated outputs) and compares every routing idea against the correct baseline: randomly mixing the cheap and the expensive model at the same accuracy.

## 2. What was delivered

- Review-1 (16/11/2026): synopsis, a 9-slide deck, a demo-guide document, and a first working two-tier router.
- A 3-model, 510-question benchmark with real per-token prices (data/benchmark_v2.json, results/run_results_v2.jsonl).
- A statistical evaluation harness: out-of-fold predictions, nested threshold selection, bootstrap confidence intervals against random mixing, paired McNemar tests.
- A frozen routing rule, validated on questions it had never seen.
- A HumanEval code benchmark (164 problems x 3 models) with a verified test-execution grader.
- A Streamlit demo (live cascade + benchmark evidence tabs).
- A reproducible repository: README, requirements, scripts organised by role.

## 3. Timeline of the work

1. **Review-1 (two tiers).** 160 questions sampled with seed 42 from MMLU and GSM8K (159 usable; one call failed). Tiers: openai/gpt-oss-20b (small) and openai/gpt-oss-120b (large) on Groq. MiniLM embeddings (384-d), XGBoost and kNN routers, 5-fold stratified cross-validation. Result: always-small 88.1% ($0.0193), always-large 91.2% ($0.0294), random 89.6% ($0.0244), kNN 88.1%, XGBoost 88.7% ($0.0195, 33.7% cheaper than always-large). The router's +0.6 point edge over always-small was not significant (bootstrap CI 0.0 to 1.9 points, McNemar p = 1.00). Reported as a null result.
2. **Why the scope grew.** The two tiers differed by only 3.1 points, so there was little for a router to win. The pool was widened with a third model (qwen/qwen3.8-27b, "premium") and the benchmark with harder questions.
3. **Benchmark v2 (510 questions).** MMLU 80, GSM8K 80, MMLU-Pro 200, MATH-500 numeric-answer subset 150. Every question answered by all three models. A resumable runner handles Groq's free-tier limits (30 requests/min, 8K tokens/min, 200K tokens/day per model, with the daily budget refilling continuously at about 8,300 tokens per hour). Each call is capped at 4096 completion tokens; truncated answers count as wrong and are reported.
4. **Analysis.** Oracle headroom, learned pre-call routers, predicted-cost routing, cascades on the small model's output length, nested threshold selection, bootstrap against random mixing, price-sensitivity sweep, and a break-even search.
5. **Rule discovery and freeze.** A learned token-count model turned out to learn mostly one fact: whether the small model hit the 4096-token cap. The plain rule "escalate to large only if small hits the cap" was frozen in git (commit e1ca13d) after the first 342 questions were labelled and before the remaining 168 were.
6. **Held-out validation.** The rule was evaluated on the 168 questions it had never seen.
7. **Code benchmark.** HumanEval on all three tiers, after the grader was verified.
8. **Demo, documentation, repository organisation.**

## 4. Final experimental setup

| tier | model | price per 1M tokens (prompt / completion) |
|---|---|---|
| small | openai/gpt-oss-20b | $0.075 / $0.30 |
| large | openai/gpt-oss-120b | $0.15 / $0.60 |
| premium | qwen/qwen3.8-27b | $0.80 / $4.00 |

A fourth model on the account (allam-2-7b) was excluded because it had no listed price.

## 5. Results

### 5.1 Main benchmark (510 questions)

| policy | accuracy | cost |
|---|---|---|
| always-small | 0.827 | $0.1321 |
| always-large | 0.886 | $0.1740 |
| always-premium | 0.812 | $0.7800 |
| oracle (any of the 3 correct) | 0.939 | - |

The premium tier is dominated: it costs about 4.5x large and is 7.4 points less accurate.

### 5.2 Learned routers (threshold chosen on training folds only, vs random mixing, points of large cost, 95% CI)

| policy | AUC | 98% accuracy target | 95% accuracy target |
|---|---|---|---|
| pre-call router (question embedding) | 0.641 | +5.3 [-2.5, +12.2] | -0.6 [-4.2, +5.7] |
| cascade on small's output length | 0.701 | +1.2 [-7.2, +12.6] | +1.4 [-5.1, +11.4] |
| cascade on embedding + length | 0.808 | -4.3 [-12.2, +6.5] | +3.6 [+0.2, +10.5] |

None of the six comparisons is robustly better than random mixing (the one interval that excludes zero has a lower bound of +0.2 and is one of six). On the earlier 342-row subset the embedding router was significantly worse than mixing at the 98% target (-6.1 [-17.8, -0.2]); with 510 rows that is no longer significant. Predicting each question's cost to improve routing did not help (cost R-squared 0.11 for small, 0.31 for large, 0.55 for premium on the 223-row interim data).

### 5.3 Where large's advantage comes from

- 31 of 510 small answers (6.1%) hit the 4096-token cap: MATH 12/150, MMLU-Pro 19/200, none on MMLU or GSM8K.
- On those, small is 0% correct and large is 84%.
- On answers that finished, small is 0.881 and large 0.889.
- So large's edge under a 4096-token cap is mostly that it finishes.

### 5.4 The cap rule

Escalate to large only if small hits the cap. Accuracy 0.878 (99.1% of always-large); paired test vs always-large: 14 vs 18 discordant questions, McNemar p = 0.60.

| large/small cost ratio | saved vs always-large | vs random mix at same accuracy (pts of large cost, 95% CI) |
|---|---|---|
| 1.3x (real) | 3.6% | +0.0 [-9.4, +11.0] |
| 1.6x | 18.8% | +13.3 [+1.6, +25.3] |
| 2.0x | 28.9% | +22.1 [+8.5, +34.9] |
| 2.6x | 41.5% | +33.1 [+16.7, +47.0] |
| 5.3x | 60.5% | +49.4 [+28.6, +65.6] |
| 21.1x | 74.8% | +61.8 [+37.2, +79.6] |

The ratio is the realized cost of large over small on this benchmark. Large's per-token price is 2x small's, but it spends fewer reasoning tokens, so the realized ratio at real prices is 1.3x. Rows with other ratios scale large's price and are hypothetical; accuracy and token counts are measured. The rule is neutral at the real ratio and clearly better than mixing from about 1.6x.

### 5.5 Held-out validation (168 questions: MATH 68, MMLU-Pro 100)

- Accuracy: always-small 0.815, always-large 0.869, cascade 0.869 (4 vs 4 discordant questions, p = 1.0).
- 11 small answers (6.5%) hit the cap; small got none right, large 82%.
- Versus random mixing: +1.1 [-14.9, +17.5] at the real ratio (1.4x on this subset), +34.9 [+11.2, +52.1] at 2.8x, +65.4 [+29.8, +82.8] at 22.1x.
- The pattern matches the development set. The held-out set has no MMLU or GSM8K rows (those were all in the first 342), so its truncation rate is higher.

### 5.6 Code (HumanEval, 164 problems)

| tier | pass | truncated | mean completion tokens | cost |
|---|---|---|---|---|
| small | 0.988 (162/164) | 2 | 647 | $0.0347 |
| large | 0.988 (162/164) | 0 | 483 | $0.0533 |
| premium | 0.957 (157/164) | 0 | 211 | $0.1625 |

Small and large pass exactly the same 162 problems; the other two fail on every model. With no accuracy gap there is nothing to route: always-small is about 35% cheaper than always-large at identical accuracy, and the cap rule saves 30.5% because it pays for two escalations that do not help. The grader was checked by running all 164 canonical solutions (all pass) and two negative controls (a stub that must fail, an infinite loop that must not pass).

## 6. What the results mean

1. Routing needs both a price gap and an accuracy gap. In this pool the price gap is small (1.3x) and the accuracy gap on knowledge and math comes almost entirely from truncation. On code there is no accuracy gap.
2. A learned pre-call router, which sees only the question text, has weak signal (AUC 0.64) and does not robustly beat random mixing.
3. A cascade pays for the small call on every question, so it beats mixing only if it escalates fewer than (1 - cost_small / cost_large) as many questions as mixing needs for the same accuracy. That is easy when the large model is clearly costlier and hard when it is barely costlier.
4. The simplest rule (escalate when the small model fails to finish) matches always-large accuracy and wins clearly once the large model costs about 1.6x or more.

## 7. Rigor

- Out-of-fold predictions everywhere; the decision threshold is chosen on training folds and scored on held-out folds.
- Bootstrap 95% intervals on the saving versus the random-mixing line, not versus the trivial endpoints.
- A rule found by inspecting the development set was frozen in git and then tested on unseen questions.
- The McNemar test against always-small is true by construction (the cascade only differs from small on truncated rows, where small is always wrong), so it is not cited as evidence.
- Truncation rates are reported and truncated answers count as wrong.
- Only logged API results are used; no number in the repository is estimated or placeholder.

## 8. Problems met and how they were handled

- Python 3.13 from python.org lacked system certificates; HTTP calls were switched to httpx/certifi.
- `load_dotenv()` failed when run from stdin; fixed by passing the path explicitly.
- Groq daily token limits stalled long runs; the runner parses the "try again in" time, waits, and resumes.
- A provider error message containing an organisation id had been saved in the results; it was removed with a same-length replacement before the first push, and a secret-scan guard runs before every commit.
- The 120B model's quota was shared with another project, which slowed the runs; the last 40 large-model HumanEval problems were run under a second Groq account. Rate limits delay calls but do not change answers.
- A first idea (predicting per-question cost) and a second (cascade on output length with a learned model) did not improve on simpler baselines; both are reported as such.

## 9. Limitations

- Accuracy is within a 4096-token budget; with a larger cap the small model might finish more questions.
- MATH-500 is limited to numeric answers (316 of 500 are numeric; 150 sampled).
- Price multipliers are hypothetical. The real price gap between the models used is small.
- The held-out set contains only MATH and MMLU-Pro questions; confidence intervals are wide at this size.
- Models ran on Groq's free tier with shared daily limits.
- The code grader runs generated code in a locked-down subprocess (timeout, memory and file limits, empty environment); it is a guard, not a hard sandbox.

## 10. Not done (future work)

- A genuine frontier-model tier that is both costlier and more accurate than the large model, which is where routing should pay most.
- Calibration analysis (ECE, Brier score) of the routers' confidence.
- A semantic caching layer before routing.
- A three-tier cascade (small, large, premium).
- Other code benchmarks and harder code tasks.

## 11. Repository map and key commits

    app.py            Streamlit demo
    src/              pipeline, evaluation runners and analysis scripts (run from the project root)
    data/             benchmark_v2.json, router_features_v2.npz, dev_ids.json (frozen development set)
    results/          raw model results (run_results_*.jsonl), sweep outputs
    review1/          first-review scripts, data and results (2 tiers, 160 questions)
    README.md, requirements.txt, PROJECT_SUMMARY.md

Key commits: 6ab9cfe (first push), 5224c09 (cascade vs router comparison, nested thresholds, price sensitivity), e1ca13d (rule frozen), 759e41c (demo), 78bb659 (layout), e2fd69d (README), 1a428cb (final results and held-out validation), b59f163 (HumanEval), a3a8761 (final file layout).

## 12. How to reproduce

    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    echo "GROQ_API_KEY=your_key" > .env
    python3 src/build_dataset_v2.py
    python3 src/run_eval_v2.py small        # also: large, premium
    python3 src/build_features_v2.py
    python3 src/compare_policies.py
    python3 src/price_sensitivity.py
    python3 src/truncation_rule.py
    python3 src/final_eval.py
    python3 src/check_harness.py
    python3 src/run_eval_code.py small      # also: large, premium
    python3 src/code_eval.py
    streamlit run app.py
