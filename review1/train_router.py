import json
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PRICE = {
    "small": {"prompt": 0.000000075, "completion": 0.0000003},
    "large": {"prompt": 0.00000015, "completion": 0.0000006},
}

data = np.load("review1/router_features.npz", allow_pickle=True)
X, y_small, y_large, ids = data["X"], data["y_small"], data["y_large"], data["ids"]

tok = {"small": {}, "large": {}}
with open("review1/run_results.jsonl") as f:
    for line in f:
        r = json.loads(line)
        if "error" in r:
            continue
        tok[r["tier"]][r["id"]] = (r["prompt_tokens"], r["completion_tokens"])

small_cost = np.array([tok["small"][i][0]*PRICE["small"]["prompt"] + tok["small"][i][1]*PRICE["small"]["completion"] for i in ids])
large_cost = np.array([tok["large"][i][0]*PRICE["large"]["prompt"] + tok["large"][i][1]*PRICE["large"]["completion"] for i in ids])

n = len(ids)
acc_always_small = y_small.mean()
acc_always_large = y_large.mean()
cost_always_small = small_cost.sum()
cost_always_large = large_cost.sum()

rng = np.random.default_rng(42)
rand_accs, rand_costs = [], []
for _ in range(1000):
    route_small = rng.random(n) < 0.5
    acc = np.where(route_small, y_small, y_large).mean()
    cost = np.where(route_small, small_cost, large_cost).sum()
    rand_accs.append(acc); rand_costs.append(cost)
acc_random, cost_random = np.mean(rand_accs), np.mean(rand_costs)

def evaluate_router(proba_small_correct):
    rows = []
    for t in np.arange(0.0, 1.01, 0.01):
        route_small = proba_small_correct >= t
        acc = np.where(route_small, y_small, y_large).mean()
        cost = np.where(route_small, small_cost, large_cost).sum()
        rows.append((t, acc, cost))
    return rows

def best_at_retention(rows, target_acc):
    candidates = [r for r in rows if r[1] >= target_acc]
    if not candidates:
        return max(rows, key=lambda r: r[1])
    return min(candidates, key=lambda r: r[2])

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "xgboost": XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=42),
    "knn_k5": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
}

print(f"n={n}  acc_always_small={acc_always_small:.3f}  acc_always_large={acc_always_large:.3f}")
print(f"cost_always_small=${cost_always_small:.6f}  cost_always_large=${cost_always_large:.6f}")
print(f"random baseline: acc={acc_random:.3f}  cost=${cost_random:.6f}")
print()

for name, clf in models.items():
    proba = cross_val_predict(clf, X, y_small, cv=skf, method="predict_proba")[:, 1]
    rows = evaluate_router(proba)
    t50 = min(rows, key=lambda r: abs(r[0]-0.5))
    r90 = best_at_retention(rows, 0.90*acc_always_large)
    r95 = best_at_retention(rows, 0.95*acc_always_large)
    print(f"--- {name} ---")
    print(f"  threshold=0.50 : acc={t50[1]:.3f} cost=${t50[2]:.6f} saved_vs_always_large={100*(1-t50[2]/cost_always_large):.1f}%")
    print(f"  @90% retention : acc={r90[1]:.3f} cost=${r90[2]:.6f} saved_vs_always_large={100*(1-r90[2]/cost_always_large):.1f}% (threshold={r90[0]:.2f})")
    print(f"  @95% retention : acc={r95[1]:.3f} cost=${r95[2]:.6f} saved_vs_always_large={100*(1-r95[2]/cost_always_large):.1f}% (threshold={r95[0]:.2f})")
    print()

import csv
proba_xgb = cross_val_predict(models["xgboost"], X, y_small, cv=skf, method="predict_proba")[:, 1]
sweep = evaluate_router(proba_xgb)
with open("review1/sweep_xgboost.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["threshold", "accuracy", "cost"])
    for t, acc, cost in sweep:
        if round(t, 2) % 0.05 == 0:
            w.writerow([round(t,2), round(acc,4), round(cost,6)])
print(open("review1/sweep_xgboost.csv").read())
