# Adaptive Cost-Aware Query Routing for LLMs

When does it pay to route a query to a cheaper LLM, and what signal should drive that decision? This project measures it end to end on real model calls (nothing simulated or estimated): three Groq-hosted models, four benchmarks, real per-token prices, and a statistical comparison against the right baseline, which is randomly mixing the cheap and the expensive model.

**Status:** complete. Main benchmark (510 questions x 3 models), held-out validation, and HumanEval (164 problems x 3 models) are all done.

## Model pool and benchmark

| tier | model | price per 1M tokens (prompt / completion) |
|---|---|---|
| small | openai/gpt-oss-20b | $0.075 / $0.30 |
| large | openai/gpt-oss-120b | $0.15 / $0.60 |
| premium | qwen/qwen3.8-27b | $0.80 / $4.00 |

- 510 questions, seed 42: MMLU 80, GSM8K 80, MMLU-Pro 200, MATH-500 numeric-answer subset 150.
- Every question is answered by every model, scored programmatically. Cost is computed from the token counts the API returns.
- Each call is capped at 4096 completion tokens. A truncated answer counts as wrong, and truncation rates are reported.

## Findings (510 questions: 80 MMLU, 80 GSM8K, 200 MMLU-Pro, 150 MATH)

| policy | accuracy | cost |
|---|---|---|
| always-small | 0.827 | $0.1321 |
| always-large | 0.886 | $0.1740 |
| always-premium | 0.812 | $0.7800 |
| oracle (any of the 3 correct) | 0.939 | - |

1. **A learned router does not robustly beat random mixing at real prices.** Correctness predictors on MiniLM question embeddings reach AUC 0.64 for the small model (0.5 is chance). Adding the small model's own output length gives 0.70 alone and 0.81 combined with the embedding. With the decision threshold chosen on training folds only, none of these policies beats randomly mixing small and large at the same accuracy: five of the six 95%/98%-target comparisons have confidence intervals that include zero, and the sixth (embedding + length cascade at the 95% target: +3.6 points of large cost, CI [+0.2, +10.5]) is not robust to multiple comparisons.
2. **Large's advantage is mostly finishing.** 31 of 510 small answers (6.1%) hit the 4096-token cap (MATH 12/150, MMLU-Pro 19/200, none on MMLU or GSM8K). Small is 0% correct on those, large is 84%. On answers that finished, small is 0.881 and large 0.889.
3. **A one-line rule captures it:** escalate to large only if small hits the cap. Accuracy 0.878, which is 99.1% of always-large (paired test vs always-large: 14 vs 18 discordant questions, McNemar p = 0.60).
4. **Whether it saves money depends on how much costlier the large model is.** The ratio below is the measured cost of large over small on this benchmark. Large's per-token price is 2x small's, but it spends fewer reasoning tokens, so the realized ratio at real prices is 1.3x. Other rows scale large's price (hypothetical; accuracy and token counts stay measured):

| large/small cost ratio | saved vs always-large | vs random mix at same accuracy (pts of large cost, 95% CI) |
|---|---|---|
| 1.3x (real) | 3.6% | +0.0 [-9.4, +11.0] |
| 1.6x | 18.8% | +13.3 [+1.6, +25.3] |
| 2.0x | 28.9% | +22.1 [+8.5, +34.9] |
| 2.6x | 41.5% | +33.1 [+16.7, +47.0] |
| 5.3x | 60.5% | +49.4 [+28.6, +65.6] |
| 21.1x | 74.8% | +61.8 [+37.2, +79.6] |

   The rule is neutral at the real 1.3x ratio and significantly better than mixing from about 1.6x.
5. **Why:** a cascade pays for the small call on every question, so it beats mixing only if it escalates fewer than (1 - cost_small / cost_large) as many questions as mixing needs for the same accuracy. That is easy when the big model is clearly costlier and hard when it is barely costlier.

### Held-out validation

The rule was frozen in git (commit e1ca13d) after the first 342 questions were labelled and before the remaining 168 were. On those 168 (MATH 68, MMLU-Pro 100; no MMLU or GSM8K rows, since all of those were in the first 342): always-small 0.815, always-large 0.869, cascade 0.869 (4 vs 4 discordant questions, p = 1.0). 11 small answers (6.5%) hit the cap; small got none right, large 82%. Versus random mixing: +1.1 [-14.9, +17.5] at the real cost ratio (1.4x on this subset), +34.9 [+11.2, +52.1] at 2.8x, +65.4 [+29.8, +82.8] at 22.1x. The pattern matches the development set.

Honesty notes:
- The cap rule was found by inspecting the development set (a learned token-count model turned out to learn mostly "hit the cap"), then frozen and tested on unseen questions.
- Price multipliers are hypothetical. Accuracy and token counts are measured.
- The held-out subset contains only MATH and MMLU-Pro, so its truncation rate is higher than the full set's.

### Code (HumanEval, 164 problems, all three tiers)

| tier | pass | truncated | mean completion tokens | cost |
|---|---|---|---|---|
| small | 0.988 (162/164) | 2 | 647 | $0.0347 |
| large | 0.988 (162/164) | 0 | 483 | $0.0533 |
| premium | 0.957 (157/164) | 0 | 211 | $0.1625 |

Small and large pass exactly the same 162 problems; the other two fail on every model (oracle = 162/164). With no accuracy gap there is nothing for a router to exploit: always-small is about 35% cheaper than always-large at identical accuracy, and the cap rule saves 30.5% because it pays for two escalations that do not help. Routing needs both a price gap and an accuracy gap. On MATH and MMLU-Pro the accuracy gap comes almost entirely from truncation; on HumanEval there is none.

Generated code runs against the official tests in a locked-down subprocess. The grader was checked by running all 164 canonical solutions (all pass) plus two negative controls (src/check_harness.py). These runs happened after the rule was frozen.

## Demo

    streamlit run app.py

Two tabs. "Live cascade": the small model answers, and the app escalates to the large model only if the small one runs out of its token budget; it shows tokens and measured cost for each step. "Benchmark evidence": the table and cost-vs-price chart above, from the logged data. A sidebar slider lowers the small model's cap for demonstration only, so an escalation can be shown live.

## Layout

    app.py            Streamlit demo
    src/              pipeline, evaluation runners and analysis scripts (run from the project root)
    data/             benchmark_v2.json, router_features_v2.npz, dev_ids.json (frozen development set)
    results/          raw model results (run_results_*.jsonl), sweep outputs
    review1/          first-review scripts, data and results (2 tiers, 160 questions)

## Reproduce

    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    echo "GROQ_API_KEY=your_key" > .env
    python3 src/build_dataset_v2.py
    python3 src/run_eval_v2.py small        # also: large, premium
    python3 src/build_features_v2.py
    python3 src/compare_policies.py     # nested threshold selection + bootstrap vs random mix
    python3 src/price_sensitivity.py
    python3 src/truncation_rule.py
    python3 src/final_eval.py           # held-out validation on rows outside data/dev_ids.json
    python3 src/check_harness.py        # grader check: canonical solutions must pass
    python3 src/run_eval_code.py small      # HumanEval; also: large, premium

## Limitations

- Accuracy is within a 4096-token budget. With a larger cap the small model might finish more questions.
- MATH-500 is limited to numeric answers (316 of 500 are numeric; 150 sampled).
- 342 development questions give wide confidence intervals.
- Models run on Groq's free tier, which has shared daily token limits.
- Generated code runs in a locked-down subprocess (timeout, memory and file limits, empty environment). This is a guard, not a hard sandbox.
- The last 40 large-model HumanEval problems were run under a different Groq account, because the free-tier quota was shared with another project. The model and prices are identical.
