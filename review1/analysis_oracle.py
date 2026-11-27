import json
from collections import defaultdict

PRICE = {  # (prompt, completion) USD per token, from Groq /models
    "small": (0.000000075, 0.0000003),
    "large": (0.00000015, 0.0000006),
    "premium": (0.0000008, 0.000004),
}
tiers = list(PRICE)
rec = defaultdict(dict)
src = {}
with open("review1/run_results.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if "error" in r:
            continue
        t = r["tier"]
        cost = r["prompt_tokens"] * PRICE[t][0] + r["completion_tokens"] * PRICE[t][1]
        rec[r["id"]][t] = {"correct": bool(r["correct"]), "cost": cost}
        src[r["id"]] = r["source"]

ids = [i for i in rec if all(t in rec[i] for t in tiers)]
print("rows with all 3 tiers valid:", len(ids))

for name, subset in [
    ("ALL", ids),
    ("MMLU", [i for i in ids if src[i] == "mmlu"]),
    ("GSM8K", [i for i in ids if src[i] == "gsm8k"]),
]:
    n = len(subset)
    print(f"\n== {name} (n={n}) ==")
    for t in tiers:
        acc = sum(rec[i][t]["correct"] for i in subset) / n
        cost = sum(rec[i][t]["cost"] for i in subset)
        print(f"{t:8s} acc={acc:.3f} cost=${cost:.4f}")

    def oracle(ts):
        ok, cost = 0, 0.0
        for i in subset:
            good = [t for t in ts if rec[i][t]["correct"]]
            if good:
                ok += 1
                cost += min(rec[i][t]["cost"] for t in good)
            else:
                cost += min(rec[i][t]["cost"] for t in ts)
        return ok / n, cost

    for label, ts in [("oracle small+large", ["small", "large"]), ("oracle all 3", tiers)]:
        acc, cost = oracle(ts)
        print(f"{label:20s} acc={acc:.3f} cost=${cost:.4f}")

    only = {t: 0 for t in tiers}
    none = 0
    for i in subset:
        good = [t for t in tiers if rec[i][t]["correct"]]
        if not good:
            none += 1
        elif len(good) == 1:
            only[good[0]] += 1
    print("questions only one tier gets right:", only, "| nobody right:", none)
