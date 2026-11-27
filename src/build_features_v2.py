import json
import numpy as np
from collections import defaultdict
from sentence_transformers import SentenceTransformer

PRICE = {  # (prompt, completion) USD per token, from Groq /models
    "small": (0.000000075, 0.0000003),
    "large": (0.00000015, 0.0000006),
    "premium": (0.0000008, 0.000004),
}
TIERS = list(PRICE)

with open("benchmark_v2.json") as f:
    rows = {r["id"]: r for r in json.load(f)}

latest = {}
with open("run_results_v2.jsonl") as f:
    for line in f:
        r = json.loads(line)
        latest[(r["id"], r["tier"])] = r  # last record wins, same as run_eval_v2

rec = defaultdict(dict)
for (rid, tier), r in latest.items():
    if "error" in r or rid not in rows:
        continue
    cost = r["prompt_tokens"] * PRICE[tier][0] + r["completion_tokens"] * PRICE[tier][1]
    rec[rid][tier] = (int(r["correct"]), cost, r["completion_tokens"])

ids = sorted(i for i in rec if all(t in rec[i] for t in TIERS))
print("rows finished on all 3 tiers:", len(ids))

model = SentenceTransformer("all-MiniLM-L6-v2")
X = model.encode([rows[i]["question"] for i in ids], batch_size=64, show_progress_bar=False)

correct = np.array([[rec[i][t][0] for t in TIERS] for i in ids])
cost = np.array([[rec[i][t][1] for t in TIERS] for i in ids])
ctok = np.array([[rec[i][t][2] for t in TIERS] for i in ids])
sources = np.array([rows[i]["source"] for i in ids])

np.savez("data/router_features_v2.npz", X=X, correct=correct, cost=cost, ctok=ctok,
         ids=np.array(ids), sources=sources, tiers=np.array(TIERS))

print("X shape:", X.shape)
for s in sorted(set(sources)):
    print(f"  {s}: {int((sources == s).sum())}")
for k, t in enumerate(TIERS):
    print(f"{t:8s} acc={correct[:, k].mean():.3f} total cost=${cost[:, k].sum():.4f}")
print(f"oracle (any tier correct): acc={correct.any(axis=1).mean():.3f}")
