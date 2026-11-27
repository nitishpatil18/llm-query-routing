import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

SEED = 42
d = np.load("data/router_features_v2.npz", allow_pickle=True)
X, correct, cost, srcs = d["X"], d["correct"], d["cost"], d["sources"]
tiers = [str(t) for t in d["tiers"]]
n, T = correct.shape
idx = np.arange(n)
mean_cost = cost.mean(axis=0)  # expected cost per question for each model
large_k = tiers.index("large")
acc_large = correct[:, large_k].mean()
cost_large = cost[:, large_k].sum()

print(f"n={n}")
for k, t in enumerate(tiers):
    print(f"always-{t:8s} acc={correct[:, k].mean():.3f} cost=${cost[:, k].sum():.4f}")
print(f"random uniform    acc={correct.mean():.3f} cost=${cost.sum() / T:.4f}")
cheapest_correct = np.where(correct.any(axis=1),
                            np.where(correct == 1, cost, np.inf).min(axis=1),
                            cost.min(axis=1))
print(f"oracle            acc={correct.any(axis=1).mean():.3f} cost=${cheapest_correct.sum():.4f}")

routers = {
    "xgboost": lambda: XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=SEED),
    "knn_k15": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=2000)),
}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
lams = np.concatenate([[0.0], np.logspace(0, 4, 400)])

for name, make in routers.items():
    P = np.column_stack([
        cross_val_predict(make(), X, correct[:, k], cv=skf, method="predict_proba")[:, 1]
        for k in range(T)
    ])
    sweep = []
    for lam in lams:
        ch = (P - lam * mean_cost[None, :]).argmax(axis=1)
        sweep.append((lam, correct[idx, ch].mean(), cost[idx, ch].sum(), ch))

    with open(f"results/sweeps/sweep_v2_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lambda", "acc", "cost"] + [f"frac_{t}" for t in tiers])
        for lam, acc, c, ch in sweep:
            w.writerow([lam, acc, c] + list(np.bincount(ch, minlength=T) / n))

    print(f"\n--- {name} ---")
    for frac in [1.0, 0.98, 0.95]:
        ok = [s for s in sweep if s[1] >= frac * acc_large - 1e-12]
        if not ok:
            print(f"  retain {frac:.0%} of always-large acc: not reached")
            continue
        lam, acc, c, ch = min(ok, key=lambda s: s[2])
        mix = np.bincount(ch, minlength=T) / n
        print(f"  retain {frac:.0%}: acc={acc:.3f} cost=${c:.4f} saved_vs_large={(1 - c / cost_large) * 100:.1f}% mix="
              + "/".join(f"{t}:{m:.2f}" for t, m in zip(tiers, mix)))
        if frac == 0.98:
            for s in sorted(set(srcs)):
                m = srcs == s
                print(f"      {s:8s} acc={correct[idx[m], ch[m]].mean():.3f} mix="
                      + "/".join(f"{t}:{(ch[m] == k).mean():.2f}" for k, t in enumerate(tiers)))
