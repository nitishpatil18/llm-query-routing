import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from xgboost import XGBClassifier
from scipy.stats import chi2

data = np.load("router_features.npz", allow_pickle=True)
X = data["X"]
y_small = data["y_small"].astype(int)
y_large = data["y_large"].astype(int)
sources = data["sources"]

THRESHOLD = 0.50
SEED = 42
n = len(y_small)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
clf = XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=SEED)
oof_proba = cross_val_predict(clf, X, y_small, cv=skf, method="predict_proba")[:, 1]

route_small = oof_proba >= THRESHOLD
router_correct = np.where(route_small, y_small, y_large)
always_small_correct = y_small
always_large_correct = y_large

print(f"n={n}")
print(f"router accuracy       = {router_correct.mean():.4f}")
print(f"always-small accuracy = {always_small_correct.mean():.4f}")
print(f"always-large accuracy = {always_large_correct.mean():.4f}")

rng = np.random.default_rng(SEED)
B = 10000
router_boot = np.empty(B)
small_boot = np.empty(B)
diff_boot = np.empty(B)
for b in range(B):
    idx = rng.integers(0, n, n)
    router_boot[b] = router_correct[idx].mean()
    small_boot[b] = always_small_correct[idx].mean()
    diff_boot[b] = router_correct[idx].mean() - always_small_correct[idx].mean()

def ci(arr):
    return np.percentile(arr, 2.5), np.percentile(arr, 97.5)

print()
print(f"router accuracy 95% CI            = {ci(router_boot)}")
print(f"always-small accuracy 95% CI      = {ci(small_boot)}")
print(f"router - always-small diff 95% CI = {ci(diff_boot)}")

b = int(((router_correct == 1) & (always_small_correct == 0)).sum())
c = int(((router_correct == 0) & (always_small_correct == 1)).sum())
if b + c > 0:
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    pval = 1 - chi2.cdf(stat, df=1)
else:
    stat, pval = 0.0, 1.0
print()
print(f"mcnemar: router-wins={b} always-small-wins={c} stat={stat:.4f} p={pval:.4f}")

clf_full = XGBClassifier(n_estimators=100, max_depth=3, eval_metric="logloss", random_state=SEED)
clf_full.fit(X, y_small)
importances = clf_full.feature_importances_
order = np.argsort(importances)[::-1]
cum = np.cumsum(importances[order])
total = importances.sum()
print()
for k in [10, 20, 50, 100]:
    print(f"top {k} embedding dims hold {100*cum[k-1]/total:.1f}% of total importance")

print()
for src in np.unique(sources):
    mask = sources == src
    print(f"{src}: router_acc={router_correct[mask].mean():.4f} "
          f"always_small_acc={always_small_correct[mask].mean():.4f} "
          f"always_large_acc={always_large_correct[mask].mean():.4f} n={int(mask.sum())}")
