# Adaptive Cost-Aware Query Routing for LLMs

When does it pay to route a query to a cheaper LLM, and what signal should drive that decision? This project measures it end to end on real model calls (nothing simulated or estimated): three Groq-hosted models, four benchmarks, real per-token prices, and a statistical comparison against the right baseline, which is randomly mixing the cheap and the expensive model.

**Status:** the development-set analysis is complete. Held-out validation (the remaining benchmark questions) and a HumanEval code benchmark are still running. Their results will be added here when they finish.

## Model pool and benchmark

| tier | model | price per 1M tokens (prompt / completion) |
|---|---|---|
| small | openai/gpt-oss-20b | $0.075 / $0.30 |
| large | openai/gpt-oss-120b | $0.15 / $0.60 |
| premium | qwen/qwen3.8-27b | $0.80 / $4.00 |

- 510 questions, seed 42: MMLU 80, GSM8K 80, MMLU-Pro 200, MATH-500 numeric-answer subset 150.
- Every question is answered by every model, scored programmatically. Cost is computed from the token counts the API returns.
- Each call is capped at 4096 completion tokens. A truncated answer counts as wrong, and truncation rates are reported.

## Findings (development set: 342 questions = 80 MMLU, 80 GSM8K, 100 MMLU-Pro, 82 MATH)

| policy | accuracy | cost |
|---|---|---|
| always-small | 0.833 | $0.0870 |
| always-large | 0.895 | $0.1118 |
| always-premium | 0.836 | $0.5231 |
| oracle (any of the 3 correct) | 0.950 | - |

1. **A learned pre-call router does not help.** Correctness predictors on MiniLM question embeddings reach AUC 0.55 for the small model (0.5 is chance). With the decision threshold chosen on training folds only, these routers are indistinguishable from, or worse than, randomly mixing small and large at the same accuracy.
2. **Large's advantage is mostly finishing.** 20 of 342 small answers (5.8%) hit the 4096-token cap (MATH 7/82, MMLU-Pro 13/100, none on MMLU or GSM8K). Small is 0% correct on those, large is 85%. On answers that finished, small is 88.5% and large 89.8%.
3. **A one-line rule captures it:** escalate to large only if small hits the cap. Accuracy 0.883, which is 98.7% of always-large. Paired test vs always-large: 10 vs 14 discordant questions, McNemar p = 0.54.
4. **Whether it saves money depends on the price gap.** At real prices large costs only 1.3x small per question here, and the saving (3.5%) is not distinguishable from mixing. If the large model were pricier (hypothetical multiplier on large's price, with measured tokens and accuracy unchanged):

| large price | large/small cost ratio | saved vs always-large | vs random mix at same accuracy (pts of large cost, 95% CI) |
|---|---|---|---|
| x1 (real) | 1.3x | 3.5% | -0.8 [-12.5, +12.8] |
| x2 | 2.6x | 42.4% | +30.4 [+12.7, +49.2] |
| x4 | 5.1x | 61.8% | +46.4 [+23.4, +68.2] |
| x16 | 20.6x | 76.4% | +58.2 [+31.0, +82.4] |

5. **Why:** a cascade pays for the small call on every question, so it beats mixing only if it escalates fewer than (1 - cost_small / cost_large) as many questions as mixing needs for the same accuracy. That is easy when the big model is several times costlier and hard when it is 1.3x.

Honesty notes:
- The cap rule was found by inspecting the development set (a learned token-count model turned out to learn mostly "hit the cap"). The rule was frozen in git (commit e1ca13d) before the remaining questions finished. Those remaining questions are the held-out test.
- Price multipliers are hypothetical. Accuracy and token counts are measured.

## Demo

    streamlit run app.py

Two tabs. "Live cascade": the small model answers, and the app escalates to the large model only if the small one runs out of its token budget; it shows tokens and measured cost for each step. "Benchmark evidence": the table and cost-vs-price chart above, from the logged data. A sidebar slider lowers the small model's cap for demonstration only, so an escalation can be shown live.

## Layout

    app.py            Streamlit demo
    src/              pipeline and analysis scripts (run from the project root)
    data/             router_features_v2.npz, dev_ids.json (frozen development set)
    results/          sweep outputs
    review1/          first-review scripts, data and results (2 tiers, 160 questions)
    run_eval_v2.py    label every question with every model (resumable)
    run_eval_code.py  HumanEval benchmark with sandboxed test execution
    benchmark_v2.json, run_results_v2.jsonl, run_results_code.jsonl   benchmark and raw results

## Reproduce

    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    echo "GROQ_API_KEY=your_key" > .env
    python3 src/build_dataset_v2.py
    python3 run_eval_v2.py small        # also: large, premium
    python3 src/build_features_v2.py
    python3 src/compare_policies.py     # nested threshold selection + bootstrap vs random mix
    python3 src/price_sensitivity.py
    python3 src/truncation_rule.py
    python3 src/final_eval.py           # held-out validation on rows outside data/dev_ids.json
    python3 src/check_harness.py        # grader check: canonical solutions must pass
    python3 run_eval_code.py small      # HumanEval; also: large, premium

## Limitations

- Accuracy is within a 4096-token budget. With a larger cap the small model might finish more questions.
- MATH-500 is limited to numeric answers (316 of 500 are numeric; 150 sampled).
- 342 development questions give wide confidence intervals.
- Models run on Groq's free tier, which has shared daily token limits.
- Generated code runs in a locked-down subprocess (timeout, memory and file limits, empty environment). This is a guard, not a hard sandbox.
